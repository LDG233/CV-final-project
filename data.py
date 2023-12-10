import os
from tqdm import tqdm
import pickle as pkl
import shutil
import argparse
import yaml
parser = argparse.ArgumentParser(description='process data')
parser.add_argument('--config', default='data.yaml', type=str, help='path to config file')
args = parser.parse_args()

class Tree:
    def __init__(self, label, parent_label='None', id=0, parent_id=0, op='none'):
        self.children = []
        self.label = label
        self.id = id
        self.parent_id = parent_id
        self.parent_label = parent_label
        self.op = op

with open(args.config, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
data_path = params['data']
if not os.path.exists(data_path):
    os.mkdir(data_path)
out_path = params['out']
if not os.path.exists(out_path):
    os.mkdir(out_path)
train_label = {}
test_label = {}
train_image = {}
test_image = {}

words_dict = ['<eos>\n','<sos>\n','struct\n']

for file_root,dirs,files in os.walk(data_path):
    for file in files:
        if file.endswith('.pkl'):
            if 'train' in file:
                with open(os.path.join(file_root,file), 'rb') as f:
                    train_image.update(pkl.load(f))
            if 'test' in file:
                with open(os.path.join(file_root,file), 'rb') as f:
                    test_image.update(pkl.load(f))
        if file == 'words_dict.txt':
            with open(os.path.join(file_root,file), 'r') as f:
                for line in f.readlines():
                    if '\n' not in line:
                        line = line + '\n'
                    if line not in words_dict:
                        words_dict.append(line)
        if file.endswith('.txt'):
            if 'train' in file:
                label_dict = train_label
            elif 'test' in file: 
                label_dict = test_label
            else:
                continue
            label = os.path.join(file_root,file)


            position = set(['^', '_'])
            math = set(['\\frac','\sqrt'])

            with open(label) as f:
                lines = f.readlines()
            num = 0
            for line in tqdm(lines):
                # line = 'RIT_2014_178.jpg x ^ { \\frac { p } { q } } = \sqrt [ q ] { x ^ { p } } = \sqrt [ q ] { x ^ { p } }'
                name, *words = line.split()
                name = name.split('.')[0]

                parents = []
                root = Tree('root', parent_label='root', parent_id=-1)

                struct_list = ['\\frac', '\sqrt']

                labels = []
                id = 1
                parents = [Tree('<sos>', id=0)]
                parent = Tree('<sos>', id=0)

                for i in range(len(words)):
                    a = words[i]
                    if a == '\\limits':
                        continue
                    if i == 0 and words[i] in ['_', '^', '{', '}']:
                        print(name)
                        break

                    elif words[i] == '{':
                        if words[i-1] == '\\frac':
                            labels.append([id, 'struct', parent.id, parent.label])
                            parents.append(Tree('\\frac', id=parent.id, op='above'))
                            id += 1
                            parent = Tree('above', id=parents[-1].id+1)
                        elif words[i-1] == '}' and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                            parent = Tree('below', id=parents[-1].id+1)
                            parents[-1].op = 'below'

                        elif words[i-1] == '\sqrt':
                            labels.append([id, 'struct', parent.id, '\sqrt'])
                            parents.append(Tree('\sqrt', id=parent.id))
                            parent = Tree('inside', id=id)
                            id += 1
                        elif words[i-1] == ']' and parents[-1].label == '\sqrt':
                            parent = Tree('inside', id=parents[-1].id+1)

                        elif words[i-1] == '^':
                            if words[i-2] != '}':
                                if words[i-2] == '\sum':
                                    labels.append([id, 'struct', parent.id, parent.label])
                                    parents.append(Tree('\sum', id=parent.id))
                                    parent = Tree('above', id=id)
                                    id += 1

                                else:
                                    labels.append([id, 'struct', parent.id, parent.label])
                                    parents.append(Tree(words[i-2], id=parent.id))
                                    parent = Tree('sup', id=id)
                                    id += 1

                            else:
                                # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                                if parents[-1].label == '\sum':
                                    parent = Tree('above', id=parents[-1].id+1)
                                else:
                                    parent = Tree('sup', id=parents[-1].id + 1)
                                # id += 1

                        elif words[i-1] == '_':
                            if words[i-2] != '}':
                                if words[i-2] == '\sum':
                                    labels.append([id, 'struct', parent.id, parent.label])
                                    parents.append(Tree('\sum', id=parent.id))
                                    parent = Tree('below', id=id)
                                    id += 1

                                else:
                                    labels.append([id, 'struct', parent.id, parent.label])
                                    parents.append(Tree(words[i-2], id=parent.id))
                                    parent = Tree('sub', id=id)
                                    id += 1

                            else:
                                # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                                if parents[-1].label == '\sum':
                                    parent = Tree('below', id=parents[-1].id+1)
                                else:
                                    parent = Tree('above', id=parents[-1].id+1)
                                # id += 1
                        else:
                            print('unknown word before {', name, i)


                    elif words[i] == '[' and words[i-1] == '\sqrt':
                        labels.append([id, 'struct', parent.id, '\sqrt'])
                        parents.append(Tree('\sqrt', id=parent.id))
                        parent = Tree('L-sup', id=id)
                        id += 1
                    elif words[i] == ']' and parents[-1].label == '\sqrt':
                        labels.append([id, '<eos>', parent.id, parent.label])
                        id += 1

                    elif words[i] == '}':

                        if words[i-1] != '}':
                            labels.append([id, '<eos>', parent.id, parent.label])
                            id += 1

                        if i + 1 < len(words) and words[i+1] == '{' and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                            continue
                        if i + 1 < len(words) and words[i + 1] in ['_', '^']:
                            continue
                        elif i + 1 < len(words) and words[i + 1] != '}':
                            parent = Tree('right', id=parents[-1].id + 1)

                        parents.pop()


                    else:
                        if words[i] in ['^', '_']:
                            continue
                        labels.append([id, words[i], parent.id, parent.label])
                        parent = Tree(words[i],id=id)
                        id += 1


                parent_dict = {0:[]}
                for i in range(len(labels)):
                    parent_dict[i+1] = []
                    parent_dict[labels[i][2]].append(labels[i][3])

                label_str = ''
                label_list = []
                for line in labels:
                    id, label, parent_id, parent_label = line
                    if label != 'struct':
                        label_str += f'{id}\t{label}\t{parent_id}\t{parent_label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n'
                        label_list.append(f'{id}\t{label}\t{parent_id}\t{parent_label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')
                    else:
                        tem = f'{id}\t{label}\t{parent_id}\t{parent_label}'
                        tem = tem + '\tabove' if 'above' in parent_dict[id] else tem + '\tNone'
                        tem = tem + '\tbelow' if 'below' in parent_dict[id] else tem + '\tNone'
                        tem = tem + '\tsub' if 'sub' in parent_dict[id] else tem + '\tNone'
                        tem = tem + '\tsup' if 'sup' in parent_dict[id] else tem + '\tNone'
                        tem = tem + '\tL-sup' if 'L-sup' in parent_dict[id] else tem + '\tNone'
                        tem = tem + '\tinside' if 'inside' in parent_dict[id] else tem + '\tNone'
                        tem = tem + '\tright' if 'right' in parent_dict[id] else tem + '\tNone'
                        label_str += tem + '\n'
                        label_list.append(tem + '\n')
                if label != '<eos>':
                    label_str += f'{id+1}\t<eos>\t{id}\t{label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n'
                    label_list.append(f'{id+1}\t<eos>\t{id}\t{label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')
                label_dict.update({name: label_list})
                for line in label_str.strip().split('\n'):
                    cid, c, pid, p, *r = line.strip().split()
                    if f'{c}\n' not in words_dict:
                        words_dict.append(f'{c}\n')
with open('./data/test/test_images.pkl', 'wb') as f:
    pkl.dump(test_image, f)
with open('./data/test/test_labels.pkl', 'wb') as f:
    pkl.dump(test_label, f)
with open('./data/train/train_images.pkl', 'wb') as f:
    pkl.dump(train_image, f)
with open('./data/train/train_labels.pkl', 'wb') as f:
    pkl.dump(train_label, f)
with open('./data/word.txt','w') as f:
    f.write(''.join(words_dict)+'above\nbelow\nsub\nsup\nL-sup\ninside\nright')