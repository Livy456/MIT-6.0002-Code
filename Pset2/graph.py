# This file contains a set of data structures to represent the graphs
# that you will be using for this pset.

class Node():
    """Represents a node in the graph"""

    def __init__(self, name):
        """
        Initializes an instance of Node object.

        Parameters:
            name: object
                The name of the node.
        """
        self.name = str(name)

    def get_name(self):
        """
        Returns:
            str
                The name of the node.
        """
        return self.name

    def __str__(self):
        """
        This is the function that is called when print(node) is called.

        Returns:
            str
                Humanly readable reprsentation of the node.
        """
        return self.name

    def __repr__(self):
        """
        Formal string representation of the node.

        Returns:
            str
                The name of the node.
        """
        return self.name

    def __eq__(self, other):
        """
        This is function called when you use the "==" operator on nodes.

        Parameters:
            other: Node
                Node object to compare against.

        Returns:
            bool
                True if self == other, False otherwise.
        """
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def __ne__(self, other):
        """
        This is function called when you used the "!=" operator on nodes.

        Parameters:
            other: Node
                Node object to compare against.

        Returns:
            bool
                True if self != other, False otherwise.
        """
        return not self.__eq__(other)

    def __hash__(self):
        """
        Returns:
            int
                Hash of the node. This function is necessary so that Nodes can be
                used as keys in a dictionary, Nodes are immutable.
        """
        return self.name.__hash__()


# PROBLEM 1: Implement this class based on the given docstring.
class DirectedRoad():
    """Represents a road (edge) with a travel time (weight)"""

    def __init__(self, src_node, dest_node, travel_time, road_type, traffic_multiplier):
        """
        Initialize src_node, dest_node, travel_time, road_type, traffic_multiplier for the DirectedRoad class

        Parameters:
            src_node: Node
                The source node.
            dest_node: Node
                The destination node.
            travel_time: float
                The time travelled between the src and dest.
            road_type: str
                The type of road of the edge.
            traffic_multiplier: float
                How much to multiply travel_time in the case of traffic.
                The traffic multiplier will always be at least 1.
        """
        # initializes DirectedRoad attributes
        self.src_node = src_node
        self.dest_node = dest_node
        self.travel_time = travel_time
        self.road_type = road_type
        self.traffic_multiplier = traffic_multiplier

    def get_source_node(self):
        """
        Getter method for DirectedRoad.

        Returns:
            Node
                The source node.
        """
        # returns the source node of the DirectedRoad object
        return self.src_node

    def get_destination_node(self):
        """
        Getter method for DirectedRoad.

        Returns:
            Node
                The destination node.
        """
        # returns the destination node of the DirectedRoad object
        return self.dest_node

    def get_road_type(self):
        """
        Getter method for DirectedRoad.

        Returns:
            str
                The road type of the road.
        """
        # returns the road type of the DirectedRoad object
        return self.road_type

    def get_travel_time(self, has_traffic=False):
        """
        Gets the travel_time for this road. If there is traffic,
        - multiply the time it takes to travel on a road by its traffic multiplier.

        Parameter:
            has_traffic: bool
                True if there is traffic, False otherwise.

        Returns:
            float
                The time to travel from the source node to the destination node.
        """
        # checks for traffic
        if has_traffic:
            # returns the travel time if traffic
            return self.travel_time * self.traffic_multiplier
        
        # returns the travel time of the DirectedRoad object
        return self.travel_time

    def get_traffic_multiplier(self):
        """
        Getter method for DirectedRoad.

        Returns:
            float
                The traffic multiplier.
        """
        # returns the traffic multiplier of the DirectedRoad object
        return self.traffic_multiplier

    def __str__(self):
        """
        Function that is called when print() is called on a DirectedRoad object.

        Returns:
            str
                With the format
                'src -> dest takes travel_time minute(s) via road_type road with traffic multiplier traffic_multiplier'

        Note: For the total time assume normal traffic conditions.
        """
        # creates a string representation of the DirectedRoad object
        DirRoadstr = f"{self.src_node} -> {self.dest_node} takes {self.travel_time} minute(s) via {self.road_type} road with traffic multiplier {self.traffic_multiplier}"
        
        # returns the string representation of the DirectedRoad object
        return DirRoadstr

    def __hash__(self):
        """
        Returns:
            int
                Hash of the road. This function is necessary so that DirectedRoads can be
                used as keys in a dictionary, DirectedRoads are immutable.
        """
        return self.__str__().__hash__()

# PROBLEM 1: Implement methods of this class based on the given docstring.
# DO NOT CHANGE THE FUNCTIONS THAT HAVE BEEN IMPLEMENTED FOR YOU.
class RoadMap():
    """Represents a road map -> a directed graph of Node and DirectedRoad objects"""

    def __init__(self):
        """
        Initalizes a new instance of RoadMap.
        """
        self.nodes = set()
        self.nodes_to_roads = {}  # must be a dictionary of Node -> list of roads starting at that node

    def __str__(self):
        """
        Function that is called when print() is called on a RoadMap object.

        Returns:
            str
                Representation of the RoadMap.
        """
        road_strs = []
        for roads in self.nodes_to_roads.values():
            for road in roads:
                road_strs.append(str(road))
        road_strs = sorted(road_strs)  # sort alphabetically
        return '\n'.join(road_strs)  # concat road_strs with "\n"s between them

    def get_all_nodes(self):
        """
        Returns:
            set of Node
                A COPY of all nodes in the RoadMap. Does not modify self.nodes.
        """
        # returns a copy of all the nodes in the RoadMap object        
        return self.nodes.copy()

    def contains_node(self, node):
        """
        Parameter:
            node: Node, node to check

        Returns:
            bool
                True, if node is in the graph; False, otherwise.
        """
        # checks if node is in set of nodes in RoadMap object
        return node in self.get_all_nodes()

    def insert_node(self, node):
        """
        Adds a Node object to the RoadMap.
        Raises a ValueError if it is already in the graph.

        Parameter:
            node: Node
                Node to add.
        """
        # checks if node is in set of nodes in RoadMap object
        if self.contains_node(node):
            raise ValueError
           
        self.nodes.add(node)               # adds a node to the set of nodes in RoadMap object 
        self.nodes_to_roads[node] = []     # maps the node to an empty list--> no road yet

    def insert_road(self, road):
        """
        Adds a DirectedRoad instance to the RoadMap.
        Raises a ValueError if either of the nodes associated with the road is not in the graph.

        Parameter:
            road: DirectedRoad
                Road to add.
        """
        src_node = road.get_source_node()          # gets the source node
        dest_node = road.get_destination_node()    # gets the destination node
        
        # checks if nodes associated with road in graph
        if src_node not in self.get_all_nodes() or dest_node not in self.get_all_nodes():
            # raise Value error if source or destination node not in roadmap
            raise ValueError
        
        self.nodes_to_roads[road.get_source_node()].append(road) # adds a DirectedRoad instance to the RoadMap
        
    def get_reachable_roads_from_node(self, node, restricted_roads):
        """
        Gets the roads out of Node node, excluding roads whose types are in restricted_roads.

        Parameters:
            node: Node
                Find reachable roads out of this node.
            restricted_roads: list of str (types of roads)
                Road types that cannot be traveled on.

        Returns:
            list of DirectedRoad
                A new list of all the roads that start at given node, whose types are not in restricted_roads.
                Empty list if the node is not in the graph.
        """
        # initialization section
        reach_roads = []
        
        # checks if node is in graph
        if node not in self.get_all_nodes():
            # returns empty list
            return []
        
        list_road = self.nodes_to_roads[node]       # all the DirectedRoad instances for specificed node
        
        # loops through all the DirectedRoad instances
        for dirRoad in list_road:
            
            # checks if road is reachable
            if dirRoad.get_road_type() not in restricted_roads:
                reach_roads.append(dirRoad)

        # returns a list of reachable DirectedRoad instances
        return reach_roads
