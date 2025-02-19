from train_agent import ACDataLoader, ActorCritic

import sys
import os
import argparse
from collections import namedtuple
import pickle as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from utils import *


logger = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class PreActorCritic(ActorCritic):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(PreActorCritic, self).__init__(state_dim, act_dim, gamma, hidden_sizes)

    def select_true_action(self, batch_state, batch_act_mask, batch_true_id, device):
        state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.ByteTensor(batch_act_mask).to(device)  # Tensor of [bs, act_dim]
        probs, value = self((state, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]

        acts = torch.LongTensor(batch_true_id).to(device)  # [bs, ]
        batch_true_probs = torch.gather(probs, 1, torch.unsqueeze(acts, 1)).squeeze()

        self.saved_actions.append(SavedAction(torch.log(batch_true_probs + 1e-9), value))
        return batch_true_id

    def update(self, optimizer, device, ent_weight):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            actor_loss += -log_prob  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        loss = actor_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item()


def pretrain(args):
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                             state_history=args.state_history)

    pretrain_path_file = '{}/uid_path_idx.pkl'.format(args.log_dir)
    uid_path_idx = pk.load(open(pretrain_path_file, 'rb'))  # [(uid, [path]), ...]
    logger.info("Load pre train path from " + pretrain_path_file)

    dataloader = ACDataLoader(uid_path_idx, args.batch_size)
    model = PreActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ### Pretrain Actor ###
    total_losses, total_plosses, total_rewards = [], [], []
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        ### Start epoch ###
        dataloader.reset()
        while dataloader.has_next():
            batch_uid_path_idx = dataloader.get_batch()
            batch_uids = [b[0] for b in batch_uid_path_idx]
            batch_path_idx = np.array([b[1] for b in batch_uid_path_idx])
            ### Start batch episodes ###
            batch_state = env.reset(batch_uids)  # numpy array of [bs, state_dim]
            act_num = 0
            done = False
            while not done:
                true_batch_act_idx = batch_path_idx[:, act_num]
                batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)  # numpy array of size [bs, act_dim]
                batch_act_idx = model.select_true_action(batch_state, batch_act_mask, true_batch_act_idx, args.device)
                batch_state, batch_reward, done = env.batch_step(batch_act_idx)
                model.rewards.append(batch_reward)
                act_num += 1
            ### End of episodes ###

            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(batch_uids) / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Update policy
            total_rewards.append(np.sum(model.rewards))
            loss, ploss = model.update(optimizer, args.device, args.ent_weight)
            total_losses.append(loss)
            total_plosses.append(ploss)
            step += 1

            # Report performance
            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                total_losses, total_plosses, total_rewards = [], [], []
                logger.info(
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss) +
                    ' | ploss={:.5f}'.format(avg_ploss) +
                    ' | reward={:.5f}'.format(avg_reward))
        ### END of epoch ###

        policy_file = '{}/pre_model_epoch_{}.ckpt'.format(args.log_dir, epoch)
        logger.info("Save pre train model to " + policy_file)
        torch.save(model.state_dict(), policy_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='pretrain_agent', help='pre train directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0.5, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/pretrain_agent_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    pretrain(args)


if __name__ == '__main__':
    main()
