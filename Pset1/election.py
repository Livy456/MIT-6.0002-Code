from state import State
import re # imports re to get rid of the \n and \t delimiters

##########################################################################################################
## Problem 1
##########################################################################################################

def load_election(filename):
    """
    Reads the contents of a file, with data given in the following tab-separated format:
    State[tab]Democrat_votes[tab]Republican_votes[tab]EC_votes

    Please ignore the first line of the file, which are the column headers, and remember that
    the special character for tab is '\t'

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a list of State instances
    """
    # initialization section
    listState = []
    
    # opens up the file
    with open(filename) as f:
        firstLine = f.readline()    # gets rid of the first line of the textfile
        stateData = f.read()        # reads the content of the file
        
    data = re.split("\n|\t", stateData) # gets rid of the newline and tab delimiters in the file
    
    # for loop to make State instances
    for i in range(len(data)):
        ele = data[i]
        
        # creates a State instance
        if ((i+1)%4 == 0) and (i != 0):
            state = data[i-3]   # gets state name
            dem = data[i-2]     # gets number of democrats in state
            rep = data[i-1]     # gets number of republicans in state
            eleCol = data[i]    # gets number of electoral colleges for state
            state = State(state, dem, rep, eleCol)  # instantiates an instance of a State
            listState.append(state) # appends the State instance to the list of states
                
    # return list of State instances
    return listState

##########################################################################################################
## Problem 2: Helper functions
##########################################################################################################
def add_ec(election):
    """"
    Adds the number of electoral colleges won by the republican and democrats
    
    Parameters:
    election- a list of state instances
    
    Return:
    a tuple, (demEc, repEc)- value of dem won electoral colleges and value of rep won electoral colleges
    
    """
    # initialization section
    demEc = 0  # number of dem won electoral colleges 
    repEc = 0  # number of rep won electoral colleges
    
    
    # loop to calculate ec for rep and dem
    for ele in election:
        # checks if democrats won the state
        if ele.get_winner() == "dem":
            demEc += ele.get_ecvotes()   # increments number of dem won ec
        
        # checks if republicans won the state    
        else:
            repEc += ele.get_ecvotes() # increments number of rep won ec
            
    # returns tuple of number of dem ec votes, number of rep ec votes    
    return (demEc, repEc)

def election_winner(election):
    """
    Finds the winner of the election based on who has the most amount of EC votes.
    Note: In this simplified representation, all of EC votes from a state go
    to the party with the majority vote.

    Parameters:
    election - a list of State instances

    Returns:
    a tuple, (winner, loser) of the election i.e. ('dem', 'rep') if Democrats won, else ('rep', 'dem')
    """
    # initialization section
    demEc = 0
    repEc = 0
    
    demEc, repEc = add_ec(election) # calculates number of electoral colleges won by dem and rep
    
    # checks if dem won election
    if demEc > repEc:
        # returns tuple of dem winning and rep losing
        return ("dem", "rep")
    # checks if rep won election
    else:
        # returns tuple of rep winning and dem losing
        return ("rep", "dem")
    

def winner_states(election):
    """
    Finds the list of States that were won by the winning candidate (lost by the losing candidate).

    Parameters:
    election - a list of State instances

    Returns:
    A list of State instances won by the winning candidate
    """
    # initialization section
    winList = []    # list of the winning state instances
    
    result = election_winner(election)      # results from election
    win = result[0] # winner of the election
    
    # finds states instances with winning poltical party
    for ele in election:
        # checks if state instance is with the winning poltical party
        if ele.get_winner() == win:
            winList.append(ele) # appends a winner state instance
                
    # returns a list of the winning state instances
    return winList

def ec_votes_to_flip(election, total=538):
    """
    Finds the number of additional EC votes required by the loser to change election outcome.
    Note: A party wins when they earn half the total number of EC votes plus 1.

    Parameters:
    election - a list of State instances
    total - total possible number of EC votes

    Returns:
    int, number of additional EC votes required by the loser to change the election outcome
    """
    #initialization section
    addEc = 0   # number of ec to add to losing candidate
    demEc = 0   # number of dem won ec
    repEc = 0   # number of rep won ec
    
    demEc, repEc = add_ec(election)     # calculates number of ec won by dem, rep
    won = election_winner(election)[0]  # finds the winner of the election
    votes = (total//2)+1    # calculates number of ec votes to win
    
    # checks if dem won
    if won == "dem":
        addEc = votes-repEc 
        
        # returns number of ec votes needed to change election
        return addEc 
    # checks if rep won
    else:
        addEc = votes-demEc
        
        # returns number of ec votes needed to change election
        return addEc

##########################################################################################################
## Problem 3: Brute Force approach
##########################################################################################################

def combinations(L):
    """
    Helper function to generate powerset of all possible combinations
    of items in input list L. E.g., if
    L is [1, 2] it will return a list with elements
    [], [1], [2], and [1,2].

    DO NOT MODIFY THIS.

    Parameters:
    L - list of items

    Returns:
    a list of lists that contains all possible
    combinations of the elements of L
    """

    def get_binary_representation(n, num_digits):
        """
        Inner function to get a binary representation of items to add to a subset,
        which combinations() uses to construct and append another item to the powerset.

        DO NOT MODIFY THIS.

        Parameters:
        n and num_digits are non-negative ints

        Returns:
            a num_digits str that is a binary representation of n
        """
        result = ''
        while n > 0:
            result = str(n%2) + result
            n = n//2
        if len(result) > num_digits:
            raise ValueError('not enough digits')
        for i in range(num_digits - len(result)):
            result = '0' + result
        return result

    powerset = []
    for i in range(0, 2**len(L)):
        binStr = get_binary_representation(i, len(L))
        subset = []
        for j in range(len(L)):
            if binStr[j] == '1':
                subset.append(L[j])
        powerset.append(subset)
    return powerset

def brute_force_swing_states(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states, these are our swing states. Iterate over
    all possible move combinations using the helper function combinations(L).
    Return the move combination that minimises the number of voters moved. If
    there exists more than one combination that minimises this, return any one of them.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances such that the election outcome would change if additional
      voters relocated to those states, as well as the number of voters required for that relocation.
    * A tuple containing the empty list followed by zero, if no possible swing states.
    """
    # initialization section
    minVote = 0
    flipStates = []     # list of state instances that flipped parties
    
    comboWin = combinations(winner_states) # gets a combination of all the winning states

    # loops over each combination of states
    for combo in comboWin:
        ec = 0
        votMov = 0  # number of votes moved
        # loops over every state instance
        for state in combo:
            margin = state.get_margin() + 1 # number of voters to flip election
            votMov+= margin                 # adds up the min number of votes needed to be moved to flip results
            ec+= state.get_ecvotes()        # calculates ec votes gained from swing state
        
        # checks if necessary votes reached
        # if previous calculation of min voters moved is still min number
        # checks if no min number of voters has been calculated
        if (ec >= ec_votes_needed) and ((votMov < minVote) or (minVote == 0)) :
            minVote = votMov # updates minimum number of voters to be moved
            flipStates = combo # updates flip state instances  
    
    # returns list of States flipped to other party, minimum number of voters to do so
    return (flipStates, minVote)

##########################################################################################################
## Problem 4: Dynamic Programming
## In this section we will define two functions, max_voters_moved and min_voters_moved, that
## together will provide a dynamic programming approach to find swing states. This problem
## is analagous to the complementary knapsack problem, you might find Lecture 1 of 6.0002 useful
## for this section of the pset.
##########################################################################################################
def max_voters_moved(winner_states, max_ec_votes):
    """
    Finds the largest number of voters needed to relocate to get at most max_ec_votes
    for the election loser.

    Analogy to the knapsack problem:
        Given a list of states each with a weight(ec_votes) and value(margin+1),
        determine the states to include in a collection so the total weight(ec_votes)
        is less than or equal to the given limit(max_ec_votes) and the total value(voters displaced)
        is as large as possible.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    max_ec_votes - int, the maximum number of EC votes

    Returns:
    * A tuple containing the list of State instances such that the maximum number of voters need to
      be relocated to these states in order to get at most max_ec_votes, and the number of voters
      required required for such a relocation.
    * A tuple containing the empty list followed by zero, if every state has a # EC votes greater
      than max_ec_votes.
    """
    # initialization section
    listState = []
    max_votMov = 0
    
    # Helper function to maximize number of voters moved from lecture 2 python file
    # to_consider --> represents election
    # avail ---> represents max_ecvotes-ec_vote(from current state instance)
    # memo --> 
    def fast_max_val(to_consider, avail, memo = None):
        """Assumes to_consider a list of subjects, avail a weight
            memo supplied by recursive calls
           Returns a tuple of the total value of a solution to the
             0/1 knapsack problem and the subjects of that solution
        Parameters:
            to_consider: list of state instances
            avail: int of available ec votes
            memo: dictionary of previously calculated values; number of state instances(key): available ec votes(value)
        """
        
        # checks if no calculations have been made
        if memo == None:
            memo = {}   # creates an empty dictionary to store 
                        # results of previous calculations for state and min voters moved
        
        # checks if calculation has already been preformed
        if (len(to_consider), avail) in memo:
            result = memo[(len(to_consider), avail)] # assigns result to previously calculated value
        
        # base case
        # checks if no more combinations or have reached the max threshold
        elif to_consider == [] or avail == 0:
            result = (0, [])
                
        # checks if state instance has too many ec votes
        # for the maximizing voters moved
        elif to_consider[0].get_ecvotes() > avail:
            #Explore right branch only
            result = fast_max_val(to_consider[1:], avail, memo)
        
        # creates branch for exploring potential combinations
        # of states to try to max voters moved, below max ec votes
        else:
            next_item = to_consider[0]
            #Explore left branch
            # with_val = number of voters moved
            # when recursively calling decrement number of available ecvotes
            with_val, with_to_take =\
                     fast_max_val(to_consider[1:],
                                avail - next_item.get_ecvotes(), memo)
            
            # num of votes moved increased
            with_val += next_item.get_margin() + 1
            #Explore right branch
            without_val, without_to_take = fast_max_val(to_consider[1:],
                                                    avail, memo)
            #Choose better branch
            if with_val > without_val:
                # with_val --> number of votes 
                # with_to_take--> list of states
                # concatentating list of states(adding next item)
                result = (with_val, with_to_take + [next_item])
            else:
                result = (without_val, without_to_take)
        # creates a new item in the dictionary 
        # make a key with the number of states we are switching and the number of ecvotes available with that many swing states
        # value is the result = > a tuple of (number of voters, list of swing states)
        memo[(len(to_consider), avail)] = result
        
        # returns a tuple of list of state instances that can be flipped, max num voters moved
        return result
    
    result = fast_max_val(winner_states, max_ec_votes)   # calls helper method to return possible
                                                         # swing states and max num of voters moved                                                     
    # returns a tuple of the list of states, and max voters moved    
    return (result[1],result[0])

def min_voters_moved(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states. Should minimize the number of voters being relocated.
    Only return states that were originally won by the winner (lost by the loser)
    of the election.

    Hint: This problem is simply the complement of max_voters_moved. You should call
    max_voters_moved with max_ec_votes set to (#ec votes won by original winner - ec_votes_needed)

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances (which we can call swing states) such that the
      minimum number of voters need to be relocated to these states in order to get at least
      ec_votes_needed, and the number of voters required for such a relocation.
    * * A tuple containing the empty list followed by zero, if no possible swing states.
    """
    #initialization section
    won_ec_votes = 0
    max_state = []      # list of win states that max voters moved 
    swing_state = [] 
    vot_mov = 0    
    
    # loops over winning states
    for win in winner_states:
        won_ec_votes+=win.get_ecvotes()     # calculates the total number of ec votes won by the winning party
                                            
    # calculates list of states instances that max num of voters moved 
    # while being below max num of ec votes needed to win                                       
    result = max_voters_moved(winner_states, won_ec_votes-ec_votes_needed) # returns a tuple
    
    max_state = result[0] # list of state instances that max num of voters moved 
    
    # loops through all states instances that were with winning party
    for state in winner_states:
        # checks to make sure state instance will min voters moved
        # checks if no possibility of swinging the election
        if (state not in max_state) and (max_state!= []):    
            swing_state.append(state) # adds state instance to list of swing state instances
        
    # loops over all the swing state instances    
    for state in swing_state:
        vot_mov += state.get_margin()+1 # sums number of voters that need to be moved  
    
    # returns list of swing state instance and the min number of voter needed to be moved
    return (swing_state, vot_mov)

##########################################################################################################
## Problem 5
##########################################################################################################
def relocate_voters(election, swing_states, ideal_states = ['AL', 'AZ', 'CA', 'TX']):
    """
    Finds a way to shuffle voters in order to flip an election outcome. Moves voters
    from states that were won by the losing candidate (states not in winner_states), to
    each of the states in swing_states. To win a swing state, you must move (margin + 1)
    new voters into that state. Any state that voters are moved from should still be won
    by the loser even after voters are moved. Also finds the number of EC votes gained by
    this rearrangement, as well as the minimum number of voters that need to be moved.
    Note: You cannot move voters out of Alabama, Arizona, California, or Texas.

    Parameters:
    election - a list of State instances representing the election
    swing_states - a list of State instances where people need to move to flip the election outcome
                   (result of min_voters_moved or brute_force_swing_states)
    ideal_states - a list of Strings holding the names of states where residents cannot be moved from
                   (default states are AL, AZ, CA, TX)

    Return:
    * A tuple that has 3 elements in the following order:
        - an int, the total number of voters moved
        - an int, the total number of EC votes gained by moving the voters
        - a dictionary with the following (key, value) mapping:
            - Key: a 2 element tuple of str, (from_state, to_state), the 2 letter State names
            - Value: int, number of people that are being moved
    * None, if it is not possible to sway the election
    """
    # initialization section
    losing_candidate_states = []
    relocate_state = {}         # dict for mapping the moving of voters from losing_state to swing_state
    index_los = 0       # index of list of losing state instances
    swing_ec = 0        # ec votes won from swing states
    tot_vot_mov = 0 
    
    winner = election_winner(election)[0]  # finds winner of the election
    
    # loops over state instances in election to find losing states(that aren't ideal states)
    for state in election:
        # checks if loser state instance and not ideal state instance
        if ((state.get_winner() != winner) and (state.get_name() not in ideal_states)):
            losing_candidate_states.append(state)
 
    count = 0

    # loops through each swing state
    for swing_state in swing_states:
        swing_win = swing_state.get_winner()
        swing_margin = swing_state.get_margin()+1
        tot_vot_mov+= swing_margin              # ec votes from swing state won
        swing_ec+= swing_state.get_ecvotes()    # swing state flipped
        
        # loops through losing states until no more losing states or swing state has been flipped
        while (index_los != len(losing_candidate_states)):
            los_state = losing_candidate_states[index_los]
            
            # checks if los state has enough votes to mov to swing state
            if los_state.get_margin()-1 - swing_margin >= 0:
                vot_mov = los_state.get_margin()-1 - (swing_state.get_margin()+1) # calculates number of voters moved
                los_state.subtract_winning_candidate_voters(vot_mov)            # moves the voters from losing state
                swing_state.add_losing_candidate_voters(vot_mov)                # adds the voters to the swing state
                relocate_state[(los_state.get_name(), swing_state.get_name())] = vot_mov    # dict (lose_state, swing_state):voters moved
                                
                count+=1
                
                # ends the loop of if the swing state has been flipped
                if (swing_state.get_winner() != swing_win):
                    break
            
            # checks if not enough voters can move from losing state to the swing state    
            else:
                vot_mov = swing_state.get_margin()+1 - (los_state.get_margin()-1) # calculates number of voters moved
                los_state.subtract_winning_candidate_voters(vot_mov)
                swing_state.add_losing_candidate_voters(vot_mov)
                index_los+=1
                
    #checks if not every swing state was flipped
    if count < len(swing_states):
        # returns None, no outcome where election is flipped
        return None
   
    # returns a tuple of the total number of voters moved, the states in which they relocated from, and num of swing ec won
    return (tot_vot_mov, swing_ec, relocate_state)
    
if __name__ == "__main__":
    pass
    # Uncomment the following lines to test each of the problems

    # # tests Problem 1
    year = 2020
    election = load_election(f"{year}_results.txt")
    #print(len(election))
    #print(election[0])

    # # tests Problem 2
    winner, loser = election_winner(election)
    won_states = winner_states(election)
    names_won_states = [state.get_name() for state in won_states]
    reqd_ec_votes = ec_votes_to_flip(election)
    print("Winner:", winner, "\nLoser:", loser)
    print("States won by the winner: ", names_won_states)
    print("EC votes needed:",reqd_ec_votes, "\n")

    # # tests Problem 3
    #brute_election = load_election("60002_results.txt")
    #brute_won_states = winner_states(brute_election)
    #brute_ec_votes_to_flip = ec_votes_to_flip(brute_election, total=14)
    #brute_swing, voters_brute = brute_force_swing_states(brute_won_states, brute_ec_votes_to_flip)
    #names_brute_swing = [state.get_name() for state in brute_swing]
    #ecvotes_brute = sum([state.get_ecvotes() for state in brute_swing])
    #print("Brute force swing states results:", names_brute_swing)
    #print("Brute force voters displaced:", voters_brute, "for a total of", ecvotes_brute, "Electoral College votes.\n")

    # # tests Problem 4a: max_voters_moved
    print("max_voters_moved")
    total_lost = sum(state.get_ecvotes() for state in won_states)
    non_swing_states, max_voters_displaced = max_voters_moved(won_states, total_lost-reqd_ec_votes)
    non_swing_states_names = [state.get_name() for state in non_swing_states]
    max_ec_votes = sum([state.get_ecvotes() for state in non_swing_states])
    print("States with the largest margins (non-swing states):", non_swing_states_names)
    print("Max voters displaced:", max_voters_displaced, "for a total of", max_ec_votes, "Electoral College votes.", "\n")

    # # tests Problem 4b: min_voters_moved
    print("min_voters_moved")
    swing_states, min_voters_displaced = min_voters_moved(won_states, reqd_ec_votes)
    # modified code starts here
    #swing_states, min_voters_displaced = min_voters_moved(won_states, reqd_ec_votes-1) # will get a tie
    # modified code ends here
    """
    - The losing party in 2020 election would only need to get AZ, GA, and WI to tie
    Needed to get 42,921 more votes which would have gotten them 37 more ec votes
    
    - If the losing party in 2020 election wants to win then they need to get AZ, GA, NV, and WI
    Needed to get 76,518 more votes which woukd have gotten them 43 more ec votes 
    """
    
    swing_state_names = [state.get_name() for state in swing_states]
    swing_ec_votes = sum([state.get_ecvotes() for state in swing_states])
    print("Complementary knapsack swing states results:", swing_state_names)
    print("Min voters displaced:", min_voters_displaced, "for a total of", swing_ec_votes, "Electoral College votes. \n")

    # # tests Problem 5: relocate_voters
    print("relocate_voters")
    flipped_election = relocate_voters(election, swing_states)
    print("Flip election mapping:", flipped_election)
