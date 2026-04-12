import multiprocessing as mp
import time

def appkit_worker():
    from AppKit import NSApplication, NSWindow, NSMakeRect, NSBackingStoreBuffered
    import objc
    app = NSApplication.sharedApplication()
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, 400, 400), 15, NSBackingStoreBuffered, False
    )
    win.makeKeyAndOrderFront_(None)
    print("Window shown on subprocess")
    # run for 2 seconds
    import Foundation
    Foundation.NSRunLoop.currentRunLoop().runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(2.0))
    print("Worker done")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    p = mp.Process(target=appkit_worker)
    p.start()
    p.join()
