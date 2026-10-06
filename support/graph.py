# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
from collections import deque
from typing import Any, Callable

Vertex = Any

class Edge:
    def __init__(self, src, tgt):
        self.source = src
        self.target = tgt
        
    def raw(self) -> tuple[Vertex, Vertex]:
        return (self.source, self.target)
    
    def flip(self):
        return Edge(self.target, self.source)
    
    def __repr__(self):
        return repr(self.raw())
    
    def __hash__(self):
        return hash(self.raw())
    
    def __eq__(self, other):
        return self.raw() == other.raw()

################################################################################
# Directed Adjacency List
################################################################################
    
class DirectedAdjList:
    def __init__(self, edge_list=[], vertex_label: Callable[[Vertex], str] | None = None,
                 vertex_text: Callable[[Vertex], str] | None = None,
                 edge_label: Callable[[Edge], str] | None = None, edge_color: Callable[[Edge], str] | None = None):
        self.out: dict[Vertex,list[Vertex]] = {}
        self.ins: dict[Vertex,list[Vertex]] = {}
        self.vertex_label = vertex_label
        self.vertex_text: Callable[[Vertex], str]
        if vertex_text is not None:
            self.vertex_text = vertex_text
        elif vertex_label is not None:
            self.vertex_text = vertex_label
        else:
            self.vertex_text = lambda v: str(v)
        self.edge_label = edge_label
        self.edge_color = edge_color
        self.edge_set: set[Edge] = set()
        for e in edge_list:
          if isinstance(e, Edge):
            self.add_edge(e.source, e.target)
          else:
            self.add_edge(e[0], e[1])

    def edges(self):
        return self.edge_set
    
    def vertices(self):
        #return range(0, self.num_vertices())
        return self.out.keys()
    
    def num_vertices(self):
        return len(self.out)
    
    def adjacent(self, u):
        self.add_vertex(u)
        return self.out[u]

    def add_vertex(self, u):
        if u not in self.out:
            self.out[u] = []
            self.ins[u] = []
    
    def add_edge(self, u, v):
        self.add_vertex(u)
        self.add_vertex(v)
        self.out[u].append(v)
        self.ins[v].append(u)
        edge = Edge(u,v)
        self.edge_set.add(edge)
        return edge

    def out_edges(self, u):
        for v in self.out[u]:
            yield Edge(u, v)
            
    def in_edges(self, v):
        for u in self.ins[v]:
            yield Edge(u, v)

    def has_edge(self, u, v):
        return Edge(u,v) in self.edge_set

    def remove_edge(self, u, v):
        self.out[u].remove(v)
        self.ins[v].remove(u)
        self.edge_set.remove(Edge(u,v))
 
    def name(self, u):
        if self.vertex_label:
            return self.vertex_label(u)
        else:
            return str(u)
    
    def named_edge(self, e):
        return (self.name(e.source), self.name(e.target))
        
    def label(self, e):
        if self.edge_label:
            return self.edge_label(e)
        else:
            return ""

    def color(self, e):
        if self.edge_color:
            return self.edge_color(e)
        else:
            return "black"
        
    def show(self, engine='neato') -> Any:
      from graphviz import Digraph #type: ignore
      dot = Digraph(engine=engine)
      for u in self.vertices():
          dot.node(self.name(u), self.vertex_text(u))
      for e in self.edges():
          dot.edge(self.name(e.source), self.name(e.target), label=self.label(e),
                   color=self.color(e), len='1.5')
      return dot

class UEdge(Edge):
    def raw(self) -> tuple[Vertex, Vertex]:
        return (self.source, self.target)
        
    def __hash__(self):
        return hash(frozenset((self.source, self.target)))
    
    def __eq__(self, other):
        return {self.source, self.target} == {other.source, other.target}


################################################################################
# Undirected Adjacency List
################################################################################

class UndirectedAdjList(DirectedAdjList):

    def add_edge(self, u, v):
        self.add_vertex(u)
        self.add_vertex(v)
        self.out[u].append(v)
        self.out[v].append(u)
        edge = UEdge(u,v)
        self.edge_set.add(edge)
        return edge

    def remove_edge(self, u, v):
        self.out[u] = [w for w in self.out[u] if w != v]
        self.out[v] = [w for w in self.out[v] if w != u]
        self.edge_set.remove(UEdge(u,v))
        
    def out_edges(self, u):
        for v in self.out[u]:
            yield UEdge(u, v)

    def in_edges(self, v):
        for u in self.out[v]:
            yield UEdge(u,v)
            
    def has_edge(self, u, v):
        return UEdge(u,v) in self.edge_set

    def show(self, engine='neato'):
        from graphviz import Graph
        dot = Graph(engine=engine)
        for u in self.vertices():
            dot.node(self.name(u))
        for e in self.edges():
            dot.edge(self.name(e.source), self.name(e.target), len='1.5')
        return dot


################################################################################
# Topological Sort
################################################################################

def topological_sort(G: DirectedAdjList) -> list[Vertex]:
    """Kahn's algorithm.  Raises ValueError if G has a cycle.

    The in-degrees are counted from the adjacency lists, not from edges(),
    because add_edge appends to the adjacency list every time but adds to
    edge_set only once: a graph with a duplicate edge disagrees with itself
    about how many predecessors a vertex has.  Counting the same way the
    loop below decrements keeps the two consistent."""
    in_degree = {u: 0 for u in G.vertices()}
    for u in G.vertices():
        for v in G.out[u]:
            in_degree[v] += 1
    queue: deque[Vertex] = deque()
    for u in G.vertices():
        if in_degree[u] == 0:
            queue.append(u)
    topo = []
    while queue:
        u = queue.pop()
        topo.append(u)
        for v in G.adjacent(u):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(topo) != G.num_vertices():
        # Every vertex left over is on, or reachable from, a cycle.
        raise ValueError('topological_sort: the graph has a cycle')
    return topo

def transpose(G: DirectedAdjList) -> DirectedAdjList:
    G_t = DirectedAdjList()
    for v in G.vertices():
        G_t.add_vertex(v)
    for e in G.edges():
        G_t.add_edge(e.target, e.source)
    return G_t
