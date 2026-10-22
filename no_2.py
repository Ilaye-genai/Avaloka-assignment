def can_attend_all(events):
    # Sort the meetings by the start time 
    # so you can compare the start time and end time
    sorted_meetings = sorted(events)
    # Get the last time that the meeting ended by 
    
    last_meeting_end_time = float("-inf")
    # Starting from the lowest possible time then update  
    # the time only if there is no overlap between the start and end time
    for start, end in sorted_meetings:
        if last_meeting_end_time > start:
            return False
        
        last_meeting_end_time = end 

    return True


import heapq

def min_rooms_required(events):
    if not events:
        return 0

    events = sorted(events)      # sort by start time
    end_times = []               # min-heap: one end time per room in use

    for start, end in events:
        if end_times and end_times[0] <= start:
            heapq.heapreplace(end_times, end)  # reuse the room that frees up first
        else:
            heapq.heappush(end_times, end)     # no free room, open a new one

    return len(end_times)
