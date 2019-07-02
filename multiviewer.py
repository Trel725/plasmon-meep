import matplotlib.pyplot as plt
# from https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
axis=-1
cb=None
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume, index_function=lambda x: x):
    global im
    global index_fun
    index_fun=index_function
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[axis] // 2
    im=ax.imshow(volume[:,:,ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    global cb
    fig = event.canvas.figure
    ax = fig.axes[0]
    if cb is not None:
        cb.remove()
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    cb=plt.colorbar(im)
    data=ax.images[0].get_array()
    im.autoscale()
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[axis]  # wrap around using %
    ax.images[0].set_array(volume[:,:,ax.index])
    ax.set_title(index_fun(ax.index))


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[axis]
    ax.images[0].set_array(volume[:,:,ax.index])
    ax.set_title(index_fun(ax.index))

