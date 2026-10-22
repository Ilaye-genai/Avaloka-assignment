class Node:
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}                                # key -> Node
        self.head, self.tail = Node(), Node()        # dummy ends
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, n):
        n.prev.next, n.next.prev = n.next, n.prev

    def _push_front(self, n):
        n.prev, n.next = self.head, self.head.next
        self.head.next.prev = n
        self.head.next = n

    def get(self, key):
        if key not in self.map:
            return -1
        n = self.map[key]
        self._remove(n)             # mark as recently used
        self._push_front(n)
        return n.val

    def put(self, key, value):
        if self.cap <= 0:
            return
        if key in self.map:         # update: drop old node
            self._remove(self.map[key])
        elif len(self.map) >= self.cap:   # full: evict LRU
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
        n = self.map[key] = Node(key, value)
        self._push_front(n)

        
