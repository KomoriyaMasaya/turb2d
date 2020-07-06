"""Utility functions for landlab grids
"""
import numpy as np
from landlab.grid.structured_quad import links
from .cip import cubic_interp_1d, rcubic_interp_1d, forester_filter, update_gradient


def set_up_neighbor_arrays(tc):
    """Create and initialize link neighbor arrays.

    Set up arrays of neighboring horizontal and vertical nodes that are
    needed for CIP solution.
    """

    # Find the neighbor nodes
    tc.core_nodes = tc.grid.core_nodes
    neighbor_nodes = tc.grid.adjacent_nodes_at_node.copy()
    tc.node_east = neighbor_nodes[:, 0]
    tc.node_north = neighbor_nodes[:, 1]
    tc.node_west = neighbor_nodes[:, 2]
    tc.node_south = neighbor_nodes[:, 3]

    # Find the neighbor links
    tc.active_links = tc.grid.active_links
    neighbor_links = links.neighbors_at_link(
        tc.grid.shape, np.arange(tc.grid.number_of_links)).copy()
    tc.link_east = neighbor_links[:, 0]
    tc.link_north = neighbor_links[:, 1]
    # This is to fix a bug in links.neighbors_at_link
    tc.link_north[tc.link_north == tc.grid.number_of_links] = -1
    tc.link_west = neighbor_links[:, 2]
    tc.link_south = neighbor_links[:, 3]

    # Find links connected to nodes, and nodes connected to links
    tc.east_link_at_node = tc.grid.links_at_node[:, 0].copy()
    tc.north_link_at_node = tc.grid.links_at_node[:, 1].copy()
    tc.west_link_at_node = tc.grid.links_at_node[:, 2].copy()
    tc.south_link_at_node = tc.grid.links_at_node[:, 3].copy()
    tc.east_link_at_node[tc.east_link_at_node == -1] = tc.west_link_at_node[
        tc.east_link_at_node == -1]
    tc.west_link_at_node[tc.west_link_at_node == -1] = tc.east_link_at_node[
        tc.west_link_at_node == -1]
    tc.north_link_at_node[tc.north_link_at_node == -1] = tc.south_link_at_node[
        tc.north_link_at_node == -1]
    tc.south_link_at_node[tc.south_link_at_node == -1] = tc.north_link_at_node[
        tc.south_link_at_node == -1]
    tc.west_node_at_horizontal_link = tc.grid.nodes_at_link[:, 0].copy()
    tc.east_node_at_horizontal_link = tc.grid.nodes_at_link[:, 1].copy()
    tc.south_node_at_vertical_link = tc.grid.nodes_at_link[:, 0].copy()
    tc.north_node_at_vertical_link = tc.grid.nodes_at_link[:, 1].copy()

    # nodes indicating outside of grids
    tc.node_east[tc.grid.nodes_at_right_edge] = tc.grid.nodes_at_right_edge
    tc.node_west[tc.grid.nodes_at_left_edge] = tc.grid.nodes_at_left_edge
    tc.node_north[tc.grid.nodes_at_top_edge] = tc.grid.nodes_at_top_edge
    tc.node_south[tc.grid.nodes_at_bottom_edge] = tc.grid.nodes_at_bottom_edge

    # Find obliquely neighbor vertical links from horizontal links
    # and obliquely neighbor horizontal links from vertical links
    tc.vertical_link_NE = tc.north_link_at_node[
        tc.east_node_at_horizontal_link]
    tc.vertical_link_SE = tc.south_link_at_node[
        tc.east_node_at_horizontal_link]
    tc.vertical_link_NW = tc.north_link_at_node[
        tc.west_node_at_horizontal_link]
    tc.vertical_link_SW = tc.south_link_at_node[
        tc.west_node_at_horizontal_link]
    tc.horizontal_link_NE = tc.east_link_at_node[
        tc.north_node_at_vertical_link]
    tc.horizontal_link_SE = tc.east_link_at_node[
        tc.south_node_at_vertical_link]
    tc.horizontal_link_NW = tc.west_link_at_node[
        tc.north_node_at_vertical_link]
    tc.horizontal_link_SW = tc.west_link_at_node[
        tc.south_node_at_vertical_link]

    # Once the neighbor arrays are set up, we change the flag to True!
    tc.neighbor_flag = True


def update_up_down_links_and_nodes(tc):
    """update location of upcurrent and downcurrent
       nodes and links
    """

    find_horizontal_up_down_nodes(tc,
                                  tc.u_node,
                                  out_up=tc.horizontal_up_nodes,
                                  out_down=tc.horizontal_down_nodes)

    find_vertical_up_down_nodes(tc,
                                tc.v_node,
                                out_up=tc.vertical_up_nodes,
                                out_down=tc.vertical_down_nodes)

    find_horizontal_up_down_links(tc,
                                  tc.u,
                                  out_up=tc.horizontal_up_links,
                                  out_down=tc.horizontal_down_links)

    find_vertical_up_down_links(tc,
                                tc.v,
                                out_up=tc.vertical_up_links,
                                out_down=tc.vertical_down_links)


def map_values(
        tc,
        h=None,
        dhdx=None,
        dhdy=None,
        u=None,
        dudx=None,
        dudy=None,
        v=None,
        dvdx=None,
        dvdy=None,
        Ch=None,
        dChdx=None,
        dChdy=None,
        eta=None,
        h_link=None,
        u_node=None,
        v_node=None,
        Ch_link=None,
        U=None,
        U_node=None,
        Kh=None,
        Cf_link=None,
        Cf_node=None,
):
    """map parameters at nodes to links, and those at links to nodes
    """
    map_links_to_nodes(
        tc,
        u=u,
        dudx=dudx,
        dudy=dudy,
        v=v,
        dvdx=dvdx,
        dvdy=dvdy,
        Cf_link=Cf_link,
        u_node=u_node,
        v_node=v_node,
        U=U,
        U_node=U_node,
        Cf_node=Cf_node,
    )
    map_nodes_to_links(tc,
                       h=h,
                       dhdx=dhdx,
                       dhdy=dhdy,
                       Ch=Ch,
                       dChdx=dChdx,
                       dChdy=dChdy,
                       eta=eta,
                       h_link=h_link,
                       Ch_link=Ch_link)


def map_links_to_nodes(tc,
                       u=None,
                       dudx=None,
                       dudy=None,
                       v=None,
                       dvdx=None,
                       dvdy=None,
                       Cf_link=None,
                       u_node=None,
                       v_node=None,
                       U=None,
                       U_node=None,
                       Cf_node=None):
    """map parameters at links to nodes
    """
    dry_links = tc.dry_links
    dry_nodes = tc.dry_nodes
    wet_pwet_nodes = tc.wet_pwet_nodes
    wet_pwet_links = tc.wet_pwet_links
    wet_pwet_vertical_links = tc.wet_pwet_vertical_links
    wet_pwet_horizontal_links = tc.wet_pwet_horizontal_links
    horizontal_link_NE = tc.horizontal_link_NE[wet_pwet_vertical_links]
    horizontal_link_NW = tc.horizontal_link_NW[wet_pwet_vertical_links]
    horizontal_link_SE = tc.horizontal_link_SE[wet_pwet_vertical_links]
    horizontal_link_SW = tc.horizontal_link_SW[wet_pwet_vertical_links]
    vertical_link_NE = tc.vertical_link_NE[wet_pwet_horizontal_links]
    vertical_link_NW = tc.vertical_link_NW[wet_pwet_horizontal_links]
    vertical_link_SE = tc.vertical_link_SE[wet_pwet_horizontal_links]
    vertical_link_SW = tc.vertical_link_SW[wet_pwet_horizontal_links]
    north_link_at_node = tc.north_link_at_node[wet_pwet_nodes]
    south_link_at_node = tc.south_link_at_node[wet_pwet_nodes]
    east_link_at_node = tc.east_link_at_node[wet_pwet_nodes]
    west_link_at_node = tc.west_link_at_node[wet_pwet_nodes]

    # set velocity zero at dry links and nodes
    if u is not None: u[dry_links] = 0
    if v is not None: v[dry_links] = 0
    if u_node is not None: u_node[dry_nodes] = 0
    if v_node is not None: v_node[dry_nodes] = 0
    if dudx is not None: dudx[dry_links] = 0
    if dudy is not None: dvdy[dry_links] = 0
    if dvdx is not None: dudx[dry_links] = 0
    if dvdy is not None: dvdy[dry_links] = 0

    # Map values of horizontal links to vertical links, and
    # values of vertical links to horizontal links.
    # Horizontal velocity is only updated at horizontal links,
    # and vertical velocity is updated at vertical links during
    # CIP procedures. Therefore we need to map those values each other.
    if u is not None:
        u[wet_pwet_vertical_links] = (
            u[horizontal_link_NE] + u[horizontal_link_NW] +
            u[horizontal_link_SE] + u[horizontal_link_SW]) / 4.0
    if v is not None:
        v[wet_pwet_horizontal_links] = (
            v[vertical_link_NE] + v[vertical_link_NW] + v[vertical_link_SE] +
            v[vertical_link_SW]) / 4.0

    # Calculate composite velocity at links
    if (U is not None) and (u is not None) and (v is not None):
        U[wet_pwet_links] = np.sqrt(u[wet_pwet_links]**2 +
                                    v[wet_pwet_links]**2)

    # map link values (u, v) to nodes
    # cubic_interp_1d(u,
    #                 dudx,
    #                 wet_pwet_nodes,
    #                 east_link_at_node,
    #                 west_link_at_node,
    #                 dx,
    #                 out=u_node)
    # cubic_interp_1d(v,
    #                 dvdy,
    #                 wet_pwet_nodes,
    #                 north_link_at_node,
    #                 south_link_at_node,
    #                 dx,
    #                 out=v_node)
    if u is not None:
        map_mean_of_links_to_node(u,
                                  wet_pwet_nodes,
                                  north_link_at_node,
                                  south_link_at_node,
                                  east_link_at_node,
                                  west_link_at_node,
                                  out=u_node)
    if v is not None:
        map_mean_of_links_to_node(v,
                                  wet_pwet_nodes,
                                  north_link_at_node,
                                  south_link_at_node,
                                  east_link_at_node,
                                  west_link_at_node,
                                  out=v_node)
    if U is not None:
        map_mean_of_links_to_node(U,
                                  wet_pwet_nodes,
                                  north_link_at_node,
                                  south_link_at_node,
                                  east_link_at_node,
                                  west_link_at_node,
                                  out=U_node)

    if (Cf_link is not None) and (Cf_node is not None):
        map_mean_of_links_to_node(Cf_link,
                                  wet_pwet_nodes,
                                  north_link_at_node,
                                  south_link_at_node,
                                  east_link_at_node,
                                  west_link_at_node,
                                  out=Cf_node)

    # tc.grid.map_mean_of_links_to_node(u, out=u_node)
    # tc.grid.map_mean_of_links_to_node(v, out=v_node)
    # tc.grid.map_mean_of_links_to_node(U, out=U_node)
    if (u_node is not None) and (u is not None):
        u_node[tc.horizontally_partial_wet_nodes] = u[
            tc.partial_wet_horizontal_links]
    if (u_node is not None) and (u is not None):
        v_node[tc.vertically_partial_wet_nodes] = v[
            tc.partial_wet_vertical_links]

    # update boundary conditions
    tc.update_boundary_conditions(
        u=u,
        v=v,
        u_node=u_node,
        v_node=v_node,
    )


def map_mean_of_links_to_node(f,
                              core,
                              north_link_at_node,
                              south_link_at_node,
                              east_link_at_node,
                              west_link_at_node,
                              out=None):

    n = f.shape[0]
    if out is None:
        out = np.zeros(n)

    out[core] = (f[north_link_at_node] + f[south_link_at_node] +
                 f[east_link_at_node] + f[west_link_at_node]) / 4.0

    return out


def map_nodes_to_links(tc,
                       h=None,
                       dhdx=None,
                       dhdy=None,
                       Ch=None,
                       dChdx=None,
                       dChdy=None,
                       eta=None,
                       h_link=None,
                       Ch_link=None):
    """map parameters at nodes to links
    """

    north_node_at_vertical_link = tc.north_node_at_vertical_link[
        tc.wet_pwet_vertical_links]
    south_node_at_vertical_link = tc.south_node_at_vertical_link[
        tc.wet_pwet_vertical_links]
    east_node_at_horizontal_link = tc.east_node_at_horizontal_link[
        tc.wet_pwet_horizontal_links]
    west_node_at_horizontal_link = tc.west_node_at_horizontal_link[
        tc.wet_pwet_horizontal_links]
    dx = tc.grid.dx

    if h is not None:
        # remove illeagal values
        adjust_negative_values(
            h,
            tc.wet_pwet_nodes,
            tc.node_east,
            tc.node_west,
            tc.node_north,
            tc.node_south,
            out_f=h,
        )
        h[tc.dry_nodes] = tc.h_init
        map_mean_of_link_nodes_to_link(h,
                                       tc.wet_pwet_horizontal_links,
                                       tc.wet_pwet_vertical_links,
                                       north_node_at_vertical_link,
                                       south_node_at_vertical_link,
                                       east_node_at_horizontal_link,
                                       west_node_at_horizontal_link,
                                       out=h_link)

    if Ch is not None:
        # remove illeagal values
        adjust_negative_values(
            Ch,
            tc.wet_pwet_nodes,
            tc.node_east,
            tc.node_west,
            tc.node_north,
            tc.node_south,
            out_f=Ch,
        )
        Ch[tc.dry_nodes] = tc.h_init * tc.C_init
        map_mean_of_link_nodes_to_link(Ch,
                                       tc.wet_pwet_horizontal_links,
                                       tc.wet_pwet_vertical_links,
                                       north_node_at_vertical_link,
                                       south_node_at_vertical_link,
                                       east_node_at_horizontal_link,
                                       west_node_at_horizontal_link,
                                       out=Ch_link)

    if dhdx is not None: dhdx[tc.dry_nodes] = 0
    if dhdy is not None: dhdy[tc.dry_nodes] = 0
    if dChdx is not None: dChdx[tc.dry_nodes] = 0
    if dChdy is not None: dChdy[tc.dry_nodes] = 0

    # map node values (h, C, eta) to links
    # rcubic_interp_1d(h,
    #                  dhdx,
    #                  tc.wet_pwet_horizontal_links,
    #                  east_node_at_horizontal_link,
    #                  west_node_at_horizontal_link,
    #                  dx,
    #                  out=h_link)
    # rcubic_interp_1d(h,
    #                  dhdy,
    #                  tc.wet_pwet_vertical_links,
    #                  north_node_at_vertical_link,
    #                  south_node_at_vertical_link,
    #                  dx,
    #                  out=h_link)
    # rcubic_interp_1d(Ch,
    #                  dChdx,
    #                  tc.wet_pwet_horizontal_links,
    #                  east_node_at_horizontal_link,
    #                  west_node_at_horizontal_link,
    #                  dx,
    #                  out=Ch_link)
    # rcubic_interp_1d(Ch,
    #                  dChdy,
    #                  tc.wet_pwet_vertical_links,
    #                  north_node_at_vertical_link,
    #                  south_node_at_vertical_link,
    #                  dx,
    #                  out=Ch_link)

    # update boundary conditions
    tc.update_boundary_conditions(h=h,
                                  Ch=Ch,
                                  h_link=h_link,
                                  Ch_link=Ch_link,
                                  eta=eta)


def map_mean_of_link_nodes_to_link(f,
                                   horizontal_link,
                                   vertical_link,
                                   north_node_at_vertical_link,
                                   south_node_at_vertical_link,
                                   east_node_at_horizontal_link,
                                   west_node_at_horizontal_link,
                                   out=None):
    """ map mean of parameters at nodes to link
    """
    if out is None:
        out = np.zeros(f.shape[0], dtype=np.int)

    out[horizontal_link] = (f[east_node_at_horizontal_link] +
                            f[west_node_at_horizontal_link]) / 2.0
    out[vertical_link] = (f[north_node_at_vertical_link] +
                          f[south_node_at_vertical_link]) / 2.0

    return out


def find_horizontal_up_down_nodes(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at horizontally upcurrent and downcurrent directions
    """
    if out_up is None:
        out_up = np.empty(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.empty(u.shape[0], dtype=np.int)

    out_up[:] = tc.node_west[:]
    out_down[:] = tc.node_east[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.node_east[negative_u_index]
    out_down[negative_u_index] = tc.node_west[negative_u_index]

    return out_up, out_down


def find_vertical_up_down_nodes(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at vertically upcurrent and downcurrent directions
    """

    if out_up is None:
        out_up = np.empty(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.empty(u.shape[0], dtype=np.int)

    out_up[:] = tc.node_south[:]
    out_down[:] = tc.node_north[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.node_north[negative_u_index]
    out_down[negative_u_index] = tc.node_south[negative_u_index]

    return out_up, out_down


def find_horizontal_up_down_links(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at horizontally upcurrent and downcurrent directions
    """
    if out_up is None:
        out_up = np.zeros(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.zeros(u.shape[0], dtype=np.int)

    out_up[:] = tc.link_west[:]
    out_down[:] = tc.link_east[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.link_east[negative_u_index]
    out_down[negative_u_index] = tc.link_west[negative_u_index]
    return out_up, out_down


def find_vertical_up_down_links(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at vertically upcurrent and downcurrent directions
    """

    if out_up is None:
        out_up = np.zeros(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.zeros(u.shape[0], dtype=np.int)

    out_up[:] = tc.link_south[:]
    out_down[:] = tc.link_north[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.link_north[negative_u_index]
    out_down[negative_u_index] = tc.link_south[negative_u_index]

    return out_up, out_down


def find_boundary_links_nodes(tc):
    """find and record boundary links and nodes
    """
    grid = tc.grid

    # Boundary Types
    FIXED_GRADIENT = grid.BC_NODE_IS_FIXED_GRADIENT
    FIXED_VALUE = grid.BC_NODE_IS_FIXED_VALUE
    CLOSED = grid.BC_NODE_IS_CLOSED

    # Find boundary and edge links
    bound_link_north = tc.south_link_at_node[grid.nodes_at_top_edge]
    bound_link_south = tc.north_link_at_node[grid.nodes_at_bottom_edge]
    bound_link_east = tc.west_link_at_node[grid.nodes_at_right_edge]
    bound_link_west = tc.east_link_at_node[grid.nodes_at_left_edge]
    edge_link_north = links.top_edge_horizontal_ids(grid.shape)
    edge_link_south = links.bottom_edge_horizontal_ids(grid.shape)
    edge_link_east = links.right_edge_vertical_ids(grid.shape)
    edge_link_west = links.left_edge_vertical_ids(grid.shape)
    tc.bound_links = np.unique(
        np.concatenate([
            bound_link_north, bound_link_south, bound_link_east,
            bound_link_west
        ]))
    tc.edge_links = np.unique(
        np.concatenate(
            [edge_link_north, edge_link_south, edge_link_east,
             edge_link_west]))

    ##################################
    # fixed gradient nodes and links #
    ##################################
    tc.fixed_grad_link_at_north = bound_link_north[grid.node_is_boundary(
        tc.north_node_at_vertical_link[bound_link_north],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_link_at_south = bound_link_south[grid.node_is_boundary(
        tc.south_node_at_vertical_link[bound_link_south],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_link_at_east = bound_link_east[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[bound_link_east],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_link_at_west = bound_link_west[grid.node_is_boundary(
        tc.west_node_at_horizontal_link[bound_link_west],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_edge_link_at_north = edge_link_north[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_north],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_edge_link_at_south = edge_link_south[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_south],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_edge_link_at_east = edge_link_east[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_east],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_edge_link_at_west = edge_link_west[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_west],
        boundary_flag=FIXED_GRADIENT)]
    tc.fixed_grad_anchor_link_at_north = tc.link_south[
        tc.fixed_grad_link_at_north]
    tc.fixed_grad_anchor_link_at_south = tc.link_north[
        tc.fixed_grad_link_at_south]
    tc.fixed_grad_anchor_link_at_east = tc.link_west[
        tc.fixed_grad_link_at_east]
    tc.fixed_grad_anchor_link_at_west = tc.link_east[
        tc.fixed_grad_link_at_west]
    tc.fixed_grad_anchor_edge_link_at_north = tc.link_south[
        tc.fixed_grad_edge_link_at_north]
    tc.fixed_grad_anchor_edge_link_at_south = tc.link_north[
        tc.fixed_grad_edge_link_at_south]
    tc.fixed_grad_anchor_edge_link_at_east = tc.link_west[
        tc.fixed_grad_edge_link_at_east]
    tc.fixed_grad_anchor_edge_link_at_west = tc.link_east[
        tc.fixed_grad_edge_link_at_west]

    # Fixed gradient nodes and adjacent nodes (anchor)
    tc.fixed_grad_nodes = grid.fixed_gradient_boundary_nodes
    tc.fixed_grad_anchor_nodes = grid.fixed_gradient_boundary_node_anchor_node

    # Fixed gradient links and adjacent links (anchor)
    tc.fixed_grad_links = np.concatenate([
        tc.fixed_grad_link_at_north, tc.fixed_grad_link_at_south,
        tc.fixed_grad_link_at_east, tc.fixed_grad_link_at_west,
        tc.fixed_grad_edge_link_at_north, tc.fixed_grad_edge_link_at_south,
        tc.fixed_grad_edge_link_at_east, tc.fixed_grad_edge_link_at_west
    ])
    tc.fixed_grad_anchor_links = np.concatenate([
        tc.fixed_grad_anchor_link_at_north, tc.fixed_grad_anchor_link_at_south,
        tc.fixed_grad_anchor_link_at_east, tc.fixed_grad_anchor_link_at_west,
        tc.fixed_grad_anchor_edge_link_at_north,
        tc.fixed_grad_anchor_edge_link_at_south,
        tc.fixed_grad_anchor_edge_link_at_east,
        tc.fixed_grad_anchor_edge_link_at_west
    ])

    ###############################
    # fixed value nodes and links #
    ###############################
    fixed_value_nodes_at_north = tc.grid.nodes_at_top_edge[
        grid.node_is_boundary(tc.grid.nodes_at_top_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_nodes_at_south = tc.grid.nodes_at_bottom_edge[
        grid.node_is_boundary(tc.grid.nodes_at_bottom_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_nodes_at_east = tc.grid.nodes_at_right_edge[
        grid.node_is_boundary(tc.grid.nodes_at_right_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_nodes_at_west = tc.grid.nodes_at_left_edge[
        grid.node_is_boundary(tc.grid.nodes_at_left_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_anchor_nodes_at_north = tc.node_south[
        fixed_value_nodes_at_north]
    fixed_value_anchor_nodes_at_south = tc.node_north[
        fixed_value_nodes_at_south]
    fixed_value_anchor_nodes_at_east = tc.node_west[fixed_value_nodes_at_east]
    fixed_value_anchor_nodes_at_west = tc.node_east[fixed_value_nodes_at_west]
    fixed_value_link_at_north = bound_link_north[grid.node_is_boundary(
        tc.north_node_at_vertical_link[bound_link_north],
        boundary_flag=FIXED_VALUE)]
    fixed_value_link_at_south = bound_link_south[grid.node_is_boundary(
        tc.south_node_at_vertical_link[bound_link_south],
        boundary_flag=FIXED_VALUE)]
    fixed_value_link_at_east = bound_link_east[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[bound_link_east],
        boundary_flag=FIXED_VALUE)]
    fixed_value_link_at_west = bound_link_west[grid.node_is_boundary(
        tc.west_node_at_horizontal_link[bound_link_west],
        boundary_flag=FIXED_VALUE)]
    fixed_value_anchor_link_at_north = tc.link_south[fixed_value_link_at_north]
    fixed_value_anchor_link_at_south = tc.link_north[fixed_value_link_at_south]
    fixed_value_anchor_link_at_east = tc.link_west[fixed_value_link_at_east]
    fixed_value_anchor_link_at_west = tc.link_east[fixed_value_link_at_west]
    fixed_value_edge_link_at_north = edge_link_north[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_north],
        boundary_flag=FIXED_VALUE)]
    fixed_value_edge_link_at_south = edge_link_south[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_south],
        boundary_flag=FIXED_VALUE)]
    fixed_value_edge_link_at_east = edge_link_east[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_east],
        boundary_flag=FIXED_VALUE)]
    fixed_value_edge_link_at_west = edge_link_west[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_west],
        boundary_flag=FIXED_VALUE)]

    # To avoid referreing indeces that do not exist
    tc.link_north[fixed_value_link_at_north] = fixed_value_link_at_north
    tc.link_south[fixed_value_link_at_south] = fixed_value_link_at_south
    tc.link_east[fixed_value_link_at_east] = fixed_value_link_at_east
    tc.link_west[fixed_value_link_at_west] = fixed_value_link_at_west

    # Record fixed value nodes and links with anchor and edge links
    tc.fixed_value_nodes = np.concatenate([
        fixed_value_nodes_at_north, fixed_value_nodes_at_south,
        fixed_value_nodes_at_east, fixed_value_nodes_at_west
    ])
    tc.fixed_value_anchor_nodes = np.concatenate([
        fixed_value_anchor_nodes_at_north, fixed_value_anchor_nodes_at_south,
        fixed_value_anchor_nodes_at_east, fixed_value_anchor_nodes_at_west
    ])
    tc.fixed_value_links = np.concatenate([
        fixed_value_link_at_north, fixed_value_link_at_south,
        fixed_value_link_at_east, fixed_value_link_at_west
    ])
    tc.fixed_value_anchor_links = np.concatenate([
        fixed_value_anchor_link_at_north, fixed_value_anchor_link_at_south,
        fixed_value_anchor_link_at_east, fixed_value_anchor_link_at_west
    ])
    tc.fixed_value_edge_links = np.concatenate([
        fixed_value_edge_link_at_north, fixed_value_edge_link_at_south,
        fixed_value_edge_link_at_east, fixed_value_edge_link_at_west
    ])


def adjust_negative_values(
        f,
        core,
        east_id,
        west_id,
        north_id,
        south_id,
        out_f=None,
        loop=1000,
):

    if out_f is None:
        out_f = f.copy()

    f_temp = f.copy()

    # counter = 0
    # to_fix = core[((out_h[core] < 0) | (out_Ch[core] < 0))]

    # while len(to_fix) > 0 and counter < loop:
    #     forester_filter(h_temp,
    #                     to_fix,
    #                     east_id,
    #                     west_id,
    #                     north_id,
    #                     south_id,
    #                     out_f=out_h)
    #     forester_filter(Ch_temp,
    #                     to_fix,
    #                     east_id,
    #                     west_id,
    #                     north_id,
    #                     south_id,
    #                     out_f=out_Ch)
    #     counter += 1
    #     to_fix = core[((out_h[core] < 0) | (out_Ch[core] < 0))]
    #     h_temp[:] = out_h[:]
    #     Ch_temp[:] = out_Ch[:]

    counter = 0
    to_fix = core[out_f[core] < 0]
    while len(to_fix) > 0 and counter < loop:
        forester_filter(f_temp,
                        to_fix,
                        east_id,
                        west_id,
                        north_id,
                        south_id,
                        out_f=out_f)
        to_fix = core[out_f[core] < 0]
        f_temp[:] = out_f[:]

    if counter == loop:
        out_f[out_f < 0] = 0
        print('Forester filter failed to fix negative values')

    return out_f
