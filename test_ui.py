import multiprocessing as mp

def ui_worker(conn):
    from AppKit import NSApplication, NSWindow, NSBackingStoreBuffered
    import Foundation
    from Foundation import NSMakeRect
    import objc
    
    app = NSApplication.sharedApplication()
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, 400, 400), 15, NSBackingStoreBuffered, False
    )
    win.makeKeyAndOrderFront_(None)
    
    # Non-blocking runloop check
    run_loop = Foundation.NSRunLoop.currentRunLoop()
    while True:
        if conn.poll():
            msg = conn.recv()
            if msg == "QUIT":
                break
        pool = objc.autorelease_pool()
        try:
            run_loop.runMode_beforeDate_(
                "kCFRunLoopDefaultMode",
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05)
            )
        finally:
            del pool

if __name__ == "__main__":
    parent, child = mp.Pipe()
    p = mp.Process(target=ui_worker, args=(child,), daemon=True)
    p.start()
    import time
    time.sleep(3)
    parent.send("QUIT")
    p.join()
    print("Success")
