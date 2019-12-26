from torch import from_numpy
import pickle as pkl

pkl_path = 'models/weights.pkl'


def load_weights_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as wp:
        name_weights = pkl.load(wp)

    print("Params load successfully !")
    return name_weights


def get_torch_state_dict(name_weights):
    state_dict = {}

    for param_name in name_weights.keys():
        state_dict['{}.weight'.format(param_name)] = from_numpy(name_weights[param_name]['weight'])
        state_dict['{}.bias'.format(param_name)] = from_numpy(name_weights[param_name]['bias'])

    return state_dict


if __name__ == '__main__':
    nw = load_weights_from_pkl(pkl_path)
    sd = get_torch_state_dict(nw)

    for k in sd.keys():
        layer = sd[k]
        print("{}: {}".format(k,layer.shape))
