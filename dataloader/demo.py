import numpy as np


def tree2dis(tree_list, shape):
    assert len(shape) == 2
    dis = np.zeros(shape)

    def dfs(tree_list):
        if not isinstance(tree_list, list):
            return {'count': 1}
        if len(tree_list) > 2:
            tree_list = [tree_list[:-1], tree_list[-1]]

        left, right = tree_list
        node = dict()
        left_node = dfs(left)
        right_node = dfs(right)
        node['count'] = left_node['count']+right_node['count']
        node['left'] = left_node
        node['right'] = right_node
        return node

    def t2d(node, dis, i, j):
        if node['count'] > 1:
            lc = node['left']['count']
            dis[i:(j+1), :] += 1
            dis[:, i:(j+1)] += 1
            dis[i:(j+1), i:(j+1)] -= 1
            dis[i:(lc+i), i:(lc+i)] = 0
            dis[(lc+i):(j+1), (lc+i):(j+1)] = 0
            t2d(node['left'], dis, i, lc+i-1)
            t2d(node['right'], dis, lc+i, j)

    root = dfs(tree_list)
    assert shape[0] == root['count']
    t2d(root, dis, 0, root['count']-1)
    return dis


if __name__ == "__main__":
    t = [1, 2, [3, 4, [5, 6], 7, [8, [9, 10]]]]
    dis = tree2dis(t, (10, 10))
    print(dis)
