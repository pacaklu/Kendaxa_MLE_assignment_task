
def to_int(lst):
    """Auxiliary function for converting list of strings to ints

    Args:
        lst (list): list of strings we want to convert

    Returns:
        list: list of integers converted from strings
    """
    for x in range(len(lst)):
        lst[x] = int(lst[x])
    return lst


def load_and_extract(path):
    """This function loads data and creates graph

    Args:
        path (string): Input path of the source file 
    
    Return:
        list: list with graph verteces
        list: list with left (red) leaves
        list: list with right leaves

    """

    with open(path, "r") as f:
        lines = f.readlines()

    graph = []
    count = 0 

    for line in lines:
        if count == 0:
            1 #continue
        
        
        elif count == (len(lines) -1):
            a = line.replace(' \n', '')
            left_leaves = a.split(' ')
            
        elif count == (len(lines) -2):
            a = line.replace(' \n', '')
            right_leaves = a.split(' ')
        else:
            line = line.replace('\n', '')
            graph.append(line.split(' ')[0])
            graph.append(line.split(' ')[1])
        count = count + 1

    graph = to_int(graph)
    right_leaves = to_int(right_leaves)
    left_leaves = to_int(left_leaves)

    return graph, right_leaves, left_leaves

def create_graph(graph):
    """Transforms graph from list structure of nodes
        to dictionary, where key would be node a values
        all its neighbours

    Args:
        graph (list): input graph as list of nodes

    Returns:
        dictionary: 'node' : [all neighbours of this node]
    """
    nodes = list(set(graph))
    
    graph_set = {}
    for node in nodes:
        set_of_neigh = []
        for x in range(len(graph)):
            if graph[x] == node:
                if x%2 == 0:
                    set_of_neigh.append(graph[x+1])
                else:
                    set_of_neigh.append(graph[x-1])
        graph_set[node] = set_of_neigh
    
    return graph_set

def dfs_paths(graph, start, goal):
    """[summary]

    Args:
        graph (dictionary): 'node' : [all neighbours of this node]
        start (int): starting node
        goal (int): goal node

    Returns:
        list: Path (list of nodes) from starting node
              to the end node
    """
    stack = [(start, [start])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == goal:
                return path
            visited.add(vertex)
            for neighbor in graph[vertex]:
                stack.append((neighbor, path + [neighbor]))


def is_subset(list1, list2):
    """check whethet list 1 is subset of list 2

    Args:
        list1 (list): list of ints
        list2 (list): list of ints

    Returns:
        int: 1 if true, 0 if false
    """
    count = 0
    
    for i in list1:
        if i in list2:
            count = count + 1
    if count == len(list1):
        return 1
    else:
        return 0  


def clear_paths(PATHS): # remove PATHS that are supersets to some another path
    """Idea is to clear from all possible paths between red and blue nodes those paths
        that are supersets to another path.
        For example if we have 2 paths, from A to B and from A to C, whereas 
        path from A to B is subset (subpath) of path from A to C, then keep 
        only path from A to B, because we can be sure that path from A to C wont
        be in final set of paths  


    Args:
        PATHS (list): list of lists (paths) between all possible pairs of blue-red nodes

    Returns:
        list: filtered list of lists (paths)
    """
    to_remove = []
    cleared = []
    for index1 in range(len(PATHS)):
        
        if PATHS[index1] in (to_remove):
            continue
        else:
            cleared.append(PATHS[index1])
            
        
        
        for index2 in range(index1+1, len(PATHS)):       
            if is_subset(PATHS[index1], PATHS[index2]) == 1:
                to_remove.append(PATHS[index2])

    return cleared

def find_unique(paths):
    """Final step, we already have list of filtered paths. 
       1. Sort paths from shortest to longest
       2. go through paths and append to list of final paths
          path, if it has no intersection from already appened
          paths

    Args:
        paths (list): list of list with paths that is filtered

    Returns:
        int: number of paths that have no intersection together
    """

    opt_paths = []
    list_of_numbers = []

    for path in paths:
        if len(path) == 1:
            opt_paths.append(path)
            list_of_numbers.extend(path)
        else:
            if len(list(set(path + list_of_numbers))) == (len(path) + len(list_of_numbers)):
                opt_paths.append(path)
                list_of_numbers.extend(path)
    
    return len(opt_paths)



def compute(path):
    """This function connects all functions together

    Args:
        path (str): Input path of the source file 

    Returns:
        integer: Desired output 
    """

    
    graph, right_leaves, left_leaves = load_and_extract(path)

    graph = create_graph(graph)

    paths = []

    for ll in left_leaves:
        for rl in right_leaves:
            part_path = dfs_paths(graph, ll, rl)
            paths.append(part_path[1:len(part_path)-1])

    paths.sort(key=len)

    #paths = clear_paths(paths)

    paths.sort(key=len)

    number_of_paths = find_unique(paths)

    return number_of_paths