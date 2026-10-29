import keyboard

stop_flag = False
skip_flag = False

def set_stop_flag(event):
    global stop_flag
    if event.event_type == 'down':
        stop_flag = True

def set_skip_flag(event):
    global skip_flag
    if event.event_type == 'down':
        skip_flag = True

keyboard.hook_key('q', set_stop_flag)
keyboard.hook_key('s', set_skip_flag)


class SafeInterruptException(Exception):
    pass
