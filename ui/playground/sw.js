/* ATOM Boss Console -- Sprint N6 service worker.
 *
 * Strategy:
 *   * Precache the static shell on install (HTML + manifest).
 *   * Network-first for navigations (so Boss always sees the latest UI
 *     while connected; falls back to cached shell when offline).
 *   * Cache-first for the manifest and SW itself.
 *   * Never cache the WebSocket or any /room/* / /metrics / /healthz
 *     endpoint -- those are live by definition.
 */

const CACHE = "atom-boss-shell-v1";
const SHELL = ["./", "./index.html", "./manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() =>
      self.skipWaiting()
    )
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Bypass everything that isn't same-origin or that is live data.
  if (url.origin !== self.location.origin) return;
  if (
    url.pathname.startsWith("/room/") ||
    url.pathname.startsWith("/metrics") ||
    url.pathname.startsWith("/healthz") ||
    url.pathname.startsWith("/api/")
  ) {
    return;
  }

  // Navigation: network first, fall back to cached shell.
  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((cache) => cache.put(event.request, copy));
          return res;
        })
        .catch(() => caches.match("./index.html"))
    );
    return;
  }

  // Static assets: cache first.
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((res) => {
        if (!res || res.status !== 200 || res.type !== "basic") return res;
        const copy = res.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        return res;
      });
    })
  );
});
