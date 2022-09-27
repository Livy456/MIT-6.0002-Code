# Finding shortest paths to drive from home to work on a road network
#
import re # imports re to help with split the delimiters from the string
from graph import DirectedRoad, Node, RoadMap


# PROBLEM 2: Building the Road Network
#
# PROBLEM 2.1: Designing your Graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the times
# represented?
#
# Write your answer below as a comment:
# The nodes represents the destinations.
# The edges represents the roads that connect the destinations(nodes) to each other. 
# The travel time is represented in Directed Road objects since
# travel time is an attribute of Directed Road object.

# PROBLEM 2.2: Implementing create_graph
def create_graph(map_filename):
    """
    Parses the map file and constructs a road map (graph).

    Travel time and traffic multiplier should be each cast to a float.

    Parameters:
        map_filename : str
            Name of the map file.

    Assumes:
        Each entry in the map file consists of the following format, separated by spaces:
            source_node destination_node travel_time road_type traffic_multiplier

        Note: hill road types always are uphill in the source to destination direction and
              downhill in the destination to the source direction. Downhill travel takes
              half as long as uphill travel. The travel_time represents the time to travel
              from source to destination (uphill).

        e.g.
            N0 N1 10 highway 1
        This entry would become two directed roads; one from 'N0' to 'N1' on a highway with
        a weight of 10.0, and another road from 'N1' to 'N0' on a highway using the same weight.

        e.g.
            N2 N3 7 uphill 2
        This entry would become two directed roads; one from 'N2' to 'N3' on a hill road with
        a weight of 7.0, and another road from 'N3' to 'N2' on a hill road with a weight of 3.5.
        Note that the directed roads created should have both type 'hill', not 'uphill'!

    Returns:
        RoadMap
            A directed road map representing the given map.
    """
    # initialization section
    sep_map_entry = []

    # instantiation section
    DirRoadMap = RoadMap()  # makes an empty road map instance
    
    # opens the file and reads the contents
    with open(map_filename) as f:
        # loops through every line of the file
        for line in f:
            # get all the attributes for the directed road object
            map_entry = line.split(" ")    
            src_node = Node(map_entry[0])    
            dest_node = Node(map_entry[1])
            travel_time = float(map_entry[2])
            road_type = map_entry[3]
            traf_mult = float(map_entry[4])
            
            # checks if road type is not a hill
            if road_type != "uphill":
                dir_road1 = DirectedRoad(src_node, dest_node, travel_time, road_type, traf_mult)    # directed road in one direction
                dir_road2 = DirectedRoad(dest_node, src_node, travel_time, road_type, traf_mult)    # directed road in other direction
            # road type is a hill
            else:
                dir_road1 = DirectedRoad(src_node, dest_node, travel_time, "hill", traf_mult)       # directed road in one direction
                dir_road2 = DirectedRoad(dest_node, src_node, travel_time/2, "hill", traf_mult)     # directed road in other direction
            
            # checks if src node not in road map
            if not DirRoadMap.contains_node(src_node):
                DirRoadMap.insert_node(src_node)
            # checks if src node not in road map
            if not DirRoadMap.contains_node(dest_node):
                DirRoadMap.insert_node(dest_node)
            
            DirRoadMap.insert_road(dir_road1)
            DirRoadMap.insert_road(dir_road2)   
            
    # returns a directed road map object, instance of RoadMap object
    return DirRoadMap

# PROBLEM 2.3: Testing create_graph
# Go to the bottom of this file, look for the section under FOR PROBLEM 2.3,
# and follow the instructions in the handout.


# PROBLEM 3: Finding the Shortest Path using Optimized Search Method
# Problem 3.1: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer:
# The objective function is to find the shortest path, which is the list
# of edges from the start node to destination node.

# The constraints are the types of roads that be traversed
# meaning edges between node for the shortest path can only 
# be created if road is not in the restricted_roads list.
# The time is our metric for selecting the shortest path.

# PROBLEM 3.2: Implement find_shortest_path
def find_shortest_path(roadmap, start, end, restricted_roads=None, has_traffic=False):
    """
    Finds the shortest path between start and end nodes on the road map,
    without using any restricted roads, following traffic conditions.
    If restricted_roads is None, assume there are no restricted roads.
    Use Dijkstra's algorithm.

    Parameters:
        roadmap: RoadMap
            The graph on which to carry out the search.
        start: Node
            Node at which to start.
        end: Node
            Node at which to end.
        restricted_roads: list of str or None
            Road Types not allowed on path. If None, all are roads allowed
        has_traffic: bool
            Flag to indicate whether to get shortest path during traffic or not.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Node, the shortest path from start to end.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    # initialization section
    best_path = []          # list of Nodes instances that form the shortest paths
    visited_nodes = {}      # calculates all the visited nodes
    time_to_node = {}       # the total travel time to all nodes
    previous_nodes = {}     # stores the previous nodes on the path
    
    # checks if start/end node in roadmap
    if start not in roadmap.get_all_nodes() or end not in roadmap.get_all_nodes():
        # no path possible, returns None
        return None
    
    # checks if the start and end node are the same
    if start == end:
        # returns tuple with starting node, travel time of 0
        return ([start], 0)
    
    unvisited_nodes = roadmap.get_all_nodes()    # gets all the nodes in roadmap
    
    # loops through all nodes and assigns the weight(travel time) to infinity
    # intializing the time travelled
    for node in roadmap.get_all_nodes():
        time_to_node[node] = float('inf') 
    time_to_node[start] = 0  # reassigns start node to have 0 travel time
    
    # loops through nodes and say none- no predeccessor node on path
    for node in roadmap.get_all_nodes():
        previous_nodes[node] = None
    
    # loops through unvisited nodes to find shortest path
    # will loop until unvisited_nodes is empty
    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda node: time_to_node[node])    # finds the min distance from previous node
        
        # no more possible paths
        if time_to_node[current_node] == float('inf'):
            break
        
        # checks if path has reached the end node
        if current_node == end:
            break   # have reached end node and gotten shortest path
        
        list_reach_road = roadmap.get_reachable_roads_from_node(current_node, restricted_roads)  # gets a list of reachable roads
        
        # loops through the list of reachable roads
        for road in list_reach_road:
            dest_node = road.get_destination_node()   # neighbor node to current node
            travel_time = road.get_travel_time(has_traffic)
            other_path = time_to_node[current_node] + travel_time   # calculates the alternative path
            # checks which neighbor node has the smallest travel time
            if other_path < time_to_node[dest_node]:
                time_to_node[dest_node] = other_path            # updates the destination node's travel time, no longer infinity
                previous_nodes[dest_node] = current_node        # adds current to the path
            
        # removes the current node due to the FIFO approach, current node now visited 
        unvisited_nodes.remove(current_node)
        
    node = end  # initializes node to start at end of the path        
    # loops through all the nodes in path    
    while previous_nodes[node] != None:
        best_path.insert(0, node)
        node = previous_nodes[node]    
        
    if best_path != []:    
        best_path.insert(0, node)
    # checks if empty path, no possible shortest path
    else:
        # no possible path from start to end node, disconnected digraph  
        return None
    # returns a tuple of the shortest path and the least amount of time
    return (best_path, time_to_node[end])

# PROBLEM 4.1: Implement optimal_path_no_traffic
def find_shortest_path_no_traffic(filename, start, end):
    """
    Finds the shortest path from start to end during conditions of no traffic.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end in normal traffic.
        If there exists no path, then return None.
    """
    graph = create_graph(filename)                          # creates a road map instance
    short_path = find_shortest_path(graph, start, end, [])  # gets a tuple of shortest path, min amount of travel time
    
    # returns the list of Node instances that create the shortest path without traffic
    return short_path[0]

# PROBLEM 4.2: Implement optimal_path_restricted
def find_shortest_path_restricted(filename, start, end):
    """
    Finds the shortest path from start to end when local roads and hill roads cannot be used.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end given the aforementioned conditions.
        If there exists no path that satisfies constraints, then return None.
    """
    graph = create_graph(filename)          # creates a road map instance
    short_path = find_shortest_path(graph, start, end, ["local", "hill"])    # gets tuple of shortest path, min amount of travel time
    # returns a list of Node instances that create the
    # shortest path while not traversing restricted roads
    return short_path[0]

# PROBLEM 4.3: Implement optimal_path_heavy_traffic
def find_shortest_path_in_traffic_no_toll(filename, start, end):
    """
    Finds the shortest path from start to end when toll roads cannot be used and in traffic,
    i.e. when all roads' travel times are multiplied by their traffic multipliers.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end given the aforementioned conditions.
        If there exists no path that satisfies the constraints, then return None.
    """
    graph = create_graph(filename)  # creates a road map instance
    short_path = find_shortest_path(graph, start, end, ["toll"], True)  # gets tuple of shortest path, min amount of travel time
    
    # returns a list of Node instances that create the
    # shortest path while not traversing restricted roads
    # and considering traffic conditions
    return short_path[0]
    
if __name__ == '__main__':

    # UNCOMMENT THE LINES BELOW TO DEBUG OR TO EXECUTE PROBLEM 2.3
    pass

    #small_map = create_graph('./maps/road_map.txt')

    # # ------------------------------------------------------------------------
    # # FOR PROBLEM 2.3
    #road_map = create_graph("maps/test_create_graph.txt")
    #print(road_map)
    # # ------------------------------------------------------------------------

    #start = Node('N0')
    #end = Node('N10')
    #restricted_roads = ["highway"]
    #print(find_shortest_path(small_map, start, end, restricted_roads))
