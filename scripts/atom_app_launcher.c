#include <Python.h>

#include <dirent.h>
#include <limits.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Must match the Python 3.9 framework used by scripts/build_atom_app_launcher.sh (Command Line Tools). */
#define ATOM_PYTHON_HOME "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9"

static int copy_parent_dir(const char *input, char *output, size_t size) {
    char buf[PATH_MAX];
    char *dir = NULL;

    if (input == NULL || output == NULL || strlen(input) >= sizeof(buf)) {
        return -1;
    }
    strcpy(buf, input);
    dir = dirname(buf);
    if (dir == NULL || strlen(dir) >= size) {
        return -1;
    }
    strcpy(output, dir);
    return 0;
}

static int should_ignore_arg(const char *arg) {
    return arg != NULL && strncmp(arg, "-psn_", 5) == 0;
}

static int discover_venv_site_packages(const char *repo_root, char *output, size_t size) {
    char lib_dir[PATH_MAX];
    DIR *dir = NULL;
    struct dirent *entry = NULL;

    if (repo_root == NULL || output == NULL) {
        return -1;
    }

    if (snprintf(lib_dir, sizeof(lib_dir), "%s/.venv/lib", repo_root) >= (int)sizeof(lib_dir)) {
        return -1;
    }

    dir = opendir(lib_dir);
    if (dir == NULL) {
        return -1;
    }

    while ((entry = readdir(dir)) != NULL) {
        char candidate[PATH_MAX];

        if (strncmp(entry->d_name, "python", 6) != 0) {
            continue;
        }

        if (snprintf(
                candidate,
                sizeof(candidate),
                "%s/%s/site-packages",
                lib_dir,
                entry->d_name
            ) >= (int)sizeof(candidate)) {
            continue;
        }

        if (access(candidate, R_OK) == 0) {
            if (strlen(candidate) >= size) {
                closedir(dir);
                return -1;
            }
            strcpy(output, candidate);
            closedir(dir);
            return 0;
        }
    }

    closedir(dir);
    return -1;
}

static wchar_t *decode_wchar_arg(const char *value) {
    if (value == NULL) {
        return NULL;
    }
    return Py_DecodeLocale(value, NULL);
}

int main(int argc, char **argv) {
    char exe_path[PATH_MAX];
    char exe_real[PATH_MAX];
    char macos_dir[PATH_MAX];
    char contents_dir[PATH_MAX];
    char app_dir[PATH_MAX];
    char repo_root[PATH_MAX];
    char venv_dir[PATH_MAX];
    char site_packages[PATH_MAX];
    char main_py[PATH_MAX];
    char pythonpath[PATH_MAX * 3];
    const char *existing_pythonpath = getenv("PYTHONPATH");
    uint32_t exe_size = sizeof(exe_path);
    int filtered_count = 0;
    int final_argc = 0;
    int arg_index = 0;
    int rc = 1;
    wchar_t **wargv = NULL;

    if (_NSGetExecutablePath(exe_path, &exe_size) != 0) {
        fprintf(stderr, "ATOM launcher failed to resolve executable path.\n");
        return 2;
    }

    if (realpath(exe_path, exe_real) == NULL) {
        strncpy(exe_real, exe_path, sizeof(exe_real) - 1);
        exe_real[sizeof(exe_real) - 1] = '\0';
    }

    if (
        copy_parent_dir(exe_real, macos_dir, sizeof(macos_dir)) != 0 ||
        copy_parent_dir(macos_dir, contents_dir, sizeof(contents_dir)) != 0 ||
        copy_parent_dir(contents_dir, app_dir, sizeof(app_dir)) != 0 ||
        copy_parent_dir(app_dir, repo_root, sizeof(repo_root)) != 0
    ) {
        fprintf(stderr, "ATOM launcher failed to derive repo root from app bundle.\n");
        return 3;
    }

    if (snprintf(venv_dir, sizeof(venv_dir), "%s/.venv", repo_root) >= (int)sizeof(venv_dir)) {
        fprintf(stderr, "ATOM launcher failed to derive virtual environment path.\n");
        return 4;
    }
    if (snprintf(main_py, sizeof(main_py), "%s/main.py", repo_root) >= (int)sizeof(main_py)) {
        fprintf(stderr, "ATOM launcher failed to derive main.py path.\n");
        return 5;
    }
    if (access(main_py, R_OK) != 0) {
        fprintf(stderr, "ATOM launcher could not find %s\n", main_py);
        return 6;
    }

    site_packages[0] = '\0';
    if (discover_venv_site_packages(repo_root, site_packages, sizeof(site_packages)) == 0) {
        if (existing_pythonpath != NULL && existing_pythonpath[0] != '\0') {
            snprintf(
                pythonpath,
                sizeof(pythonpath),
                "%s:%s:%s",
                repo_root,
                site_packages,
                existing_pythonpath
            );
        } else {
            snprintf(
                pythonpath,
                sizeof(pythonpath),
                "%s:%s",
                repo_root,
                site_packages
            );
        }
    } else if (existing_pythonpath != NULL && existing_pythonpath[0] != '\0') {
        snprintf(
            pythonpath,
            sizeof(pythonpath),
            "%s:%s",
            repo_root,
            existing_pythonpath
        );
    } else {
        snprintf(pythonpath, sizeof(pythonpath), "%s", repo_root);
    }

    setenv("PYTHONPATH", pythonpath, 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("VIRTUAL_ENV", venv_dir, 1);
    setenv("ATOM_APP_BUNDLE", app_dir, 1);
    setenv("ATOM_LAUNCHED_FROM_APP", "1", 1);
    setenv("ATOM_LAUNCH_MODE", "bundle", 1);
    setenv("PYTHONHOME", ATOM_PYTHON_HOME, 1);
    chdir(repo_root);

    for (arg_index = 1; arg_index < argc; ++arg_index) {
        if (!should_ignore_arg(argv[arg_index])) {
            filtered_count += 1;
        }
    }

    final_argc = (filtered_count == 0) ? 2 : (filtered_count + 1);
    wargv = (wchar_t **)calloc((size_t)final_argc + 1, sizeof(wchar_t *));
    if (wargv == NULL) {
        fprintf(stderr, "ATOM launcher could not allocate Python argv.\n");
        return 7;
    }

    wargv[0] = decode_wchar_arg(exe_real);
    if (wargv[0] == NULL) {
        fprintf(stderr, "ATOM launcher could not decode executable path.\n");
        free(wargv);
        return 8;
    }

    if (filtered_count == 0) {
        wargv[1] = decode_wchar_arg(main_py);
        if (wargv[1] == NULL) {
            fprintf(stderr, "ATOM launcher could not decode main.py path.\n");
            PyMem_RawFree(wargv[0]);
            free(wargv);
            return 9;
        }
    } else {
        int final_index = 1;
        for (arg_index = 1; arg_index < argc; ++arg_index) {
            if (should_ignore_arg(argv[arg_index])) {
                continue;
            }
            wargv[final_index] = decode_wchar_arg(argv[arg_index]);
            if (wargv[final_index] == NULL) {
                fprintf(stderr, "ATOM launcher could not decode argument %s\n", argv[arg_index]);
                for (int i = 0; i < final_index; ++i) {
                    PyMem_RawFree(wargv[i]);
                }
                free(wargv);
                return 10;
            }
            final_index += 1;
        }
    }

    {
        wchar_t *py_home_w = Py_DecodeLocale(ATOM_PYTHON_HOME, NULL);
        if (py_home_w != NULL) {
            Py_SetPythonHome(py_home_w);
        }
    }
    Py_SetProgramName(wargv[0]);
    rc = Py_Main(final_argc, wargv);

    for (arg_index = 0; arg_index < final_argc; ++arg_index) {
        if (wargv[arg_index] != NULL) {
            PyMem_RawFree(wargv[arg_index]);
        }
    }
    free(wargv);
    return rc;
}
