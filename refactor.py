import os

main_path = r'c:\Users\satyamy\Perforce\Tools\ATOM\main.py'
wiring_path = r'c:\Users\satyamy\Perforce\Tools\ATOM\core\boot\wiring.py'

with open(main_path, 'r', encoding='utf-8') as f:
    full_text = f.read()

start_idx = full_text.find('def _wire_events(')
end_str = 'async def main() -> None:'
end_idx = full_text.find(end_str)

if start_idx != -1 and end_idx != -1:
    # Safely backtrack to the actual end of _wire_events
    # It returns a dict. The last line is "    }"
    
    wire_block_raw = full_text[start_idx:end_idx]
    
    # Prepend imports to wiring.py
    imports = '''"""
ATOM -- Core Event Wiring

Extracted event bus attachments from the main entry point.
"""
from __future__ import annotations
import asyncio
import logging
import time

logger = logging.getLogger("atom.wiring")

'''

    wiring_text = imports + wire_block_raw.replace('def _wire_events(', 'def wire_events(').replace('logger = logging.getLogger("atom.main")', '')

    with open(wiring_path, 'w', encoding='utf-8') as f:
        f.write(wiring_text)
        
    # Replace in main.py
    new_main_text = full_text[:start_idx] + 'from core.boot.wiring import wire_events\n\n\n' + full_text[end_idx:]
    
    # Fix the call site from _wire_events to wire_events
    new_main_text = new_main_text.replace('_wiring_ctx = _wire_events(', '_wiring_ctx = wire_events(')
    
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(new_main_text)
    
    print('Extraction of wire_events successful.')
else:
    print('Failed to locate _wire_events bounds.')
