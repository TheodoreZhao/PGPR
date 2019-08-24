import os
import argparse
import pickle as pk

from tqdm import tqdm

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from utils import *


logger = None


class MyBatchKGEnvironment(BatchKGEnvironment):
    def __init__(self, dataset_str, max_acts, max_path_len=3, state_history=1):
        super(MyBatchKGEnvironment, self).__init__(dataset_str, max_acts, max_path_len, state_history)

    def batch_search_idx_from_action(self, batch_act):
        """
        Args:
            batch_act: list of actions （relation, node_type, node_id）.
        Returns:
            batch_act_idx: list of integers.
        """
        batch_act_idx = []
        false_b = []
        for i in range(len(batch_act)):
            act = (batch_act[i][0], batch_act[i][2])
            action_space = self._batch_curr_actions[i]
            if act in action_space:
                batch_act_idx.append(action_space.index(act))
            else:
                batch_act_idx.append(0)
                false_b.append(i)
        return batch_act_idx, false_b


def heuristic_meta_path(args, train_user_products):
    env = MyBatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    kg = env.kg
    assert isinstance(kg, KnowledgeGraph)
    uids = list(train_user_products.keys())
    pattern_ids = PATH_PATTERN.keys()  # all patterns
    pretrain_paths = []
    for u in tqdm(uids):
        pids = train_user_products[u]
        for p in pids:
            for pattern_id in pattern_ids:
                paths = kg.heuristic_search(u, p, pattern_id=pattern_id)
                paths = [list(path) for path in paths]  # [[e1, e2, e3, e4], ...]
                pattern = PATH_PATTERN[pattern_id]
                pattern = [(SELF_LOOP, USER)] + [i for i in pattern[1:]]    # [(SELF_LOOP, USER),(PURCHASE, PRODUCT),(PURCHASE, USER),(PURCHASE, PRODUCT)]
                if pattern_id == 1:
                    pattern.append((SELF_LOOP, PRODUCT))
                    paths = [path + [path[-1]] for path in paths]
                for i in range(len(paths)):
                    for j in range(len(paths[i])):
                        paths[i][j] = list(pattern[j]) + [paths[i][j]]  # [[(relation, node_type, node_id), ...], ...]
                pretrain_paths.extend(paths)

    # convert to action idx
    batch_uids = [path[0][-1] for path in pretrain_paths]
    valid_path = [True] * len(batch_uids)
    act_idx = []
    ### Start batch episodes ###
    batch_state = env.reset(batch_uids)  # numpy array of [bs, state_dim]
    done = False
    action_num = 0
    while not done:
        batch_action = [path[action_num+1] for path in pretrain_paths]
        batch_act_idx, false_b = env.batch_search_idx_from_action(batch_action)
        batch_state, _, done = env.batch_step(batch_act_idx)
        for b in false_b:
            valid_path[b] = False
        act_idx.append(batch_act_idx)
        action_num += 1
    ### End of episodes ###
    act_idx = np.transpose(np.array(act_idx, dtype=int))
    act_idx = list(act_idx)
    true_path = []
    for i, valid in enumerate(valid_path):
        if valid:
            uid = batch_uids[i]
            act_idx_path = act_idx[i]
            true_path.append((uid, act_idx_path))

    pretrain_path_file = '{}/uid_path_idx.pkl'.format(args.log_dir)
    pk.dump(true_path, open(pretrain_path_file, 'wb'))
    logger.info("Save pre train path to " + pretrain_path_file)


def heuristic_shortest_path(args, train_user_products):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--type', type=str, default='meta', help='type of generate heuristic path: {meta, shortest}')
    parser.add_argument('--name', type=str, default='pretrain_agent', help='pre train directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')

    args = parser.parse_args()

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/path_generate_log.txt')
    logger.info(args)

    train_labels = load_labels(args.dataset, 'train')  # {uid:[pid], ...}

    set_random_seed(args.seed)
    if args.type == 'meta':
        heuristic_meta_path(args, train_labels)
    elif args.type == 'shortest':
        heuristic_shortest_path(args, train_labels)
    else:
        return 0


if __name__ == '__main__':
    main()
