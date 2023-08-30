

def load_default_gap(name, num):
    if name == 'Thursday-01-03':
        gap = 0.1

    elif name == 'cover':
        if num == 4:
            gap = 0.6
        if num == 8:
            gap = 0.4
        else:
            gap = 0.2

    elif name == 'PageBlock':
        gap = 0.3

    elif name == 'Shuttle':
        gap = 0.4

    else:
        gap = 2

    return gap
