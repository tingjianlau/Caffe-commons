import pydot
edg = [(1,2),(1,3)]
g = pydot.graph_form_edges(edg)
g.write_jpeg('graph.jpg', prog='dot')
