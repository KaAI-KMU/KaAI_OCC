import pickle

def load_and_inspect_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

# train_pkl_path = './data/nuscenes/bevdetv2-nuscenes_infos_train.pkl'
# val_pkl_path = './data/nuscenes/bevdetv2-nuscenes_infos_val.pkl'

val_pkl_path = './data/nuscenes/nuscenes_infos_val_occ.pkl'
train_pkl_path = './data/nuscenes/nuscenes_infos_train_occ.pkl'

# Load and inspect the train set
train_data = load_and_inspect_pkl(train_pkl_path)['infos']
print("hello train")


# val_data = load_and_inspect_pkl(val_pkl_path)['infos']
# val_pkl = load_and_inspect_pkl(val_pkl_path)
# val_pkl['infos'] = val_pkl['infos'][:1000]

# val100_save_path = './data/nuscenes/bevdetv2-nuscenes_infos_val1000.pkl'
# with open(val100_save_path, 'wb') as f:
#     pickle.dump(val_pkl, f)
val_data = load_and_inspect_pkl(val_pkl_path)

print("hello val")

