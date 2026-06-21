import rdflib
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import os
import random
import numpy as np
from collections import defaultdict


def shorten_uri(uri):
    """
    Shorten a URI by extracting the fragment or the last part of the path
    """
    if '#' in uri:
        return uri.split('#')[-1]
    elif '/' in uri:
        return uri.split('/')[-1]
    else:
        return uri


def filter_triples(graph, predicate_filter=None, subject_filter=None, object_filter=None):
    """
    Filter RDF triples based on various criteria

    Args:
        graph: RDFlib graph object
        predicate_filter: string to filter predicates (e.g., 'type', 'label')
        subject_filter: string to filter subjects
        object_filter: string to filter objects

    Returns:
        Filtered RDFlib graph
    """
    filtered_graph = rdflib.Graph()

    for s, p, o in graph:
        include = True

        # Filter by predicate
        if predicate_filter and predicate_filter not in str(p):
            include = False

        # Filter by subject
        if subject_filter and subject_filter not in str(s):
            include = False

        # Filter by object
        if object_filter and object_filter not in str(o):
            include = False

        if include:
            filtered_graph.add((s, p, o))

    return filtered_graph


def find_connected_components(graph):
    """
    Find connected components in the RDF graph
    Returns list of subgraphs (connected components)
    """
    nx_graph = nx.DiGraph()

    # Convert to NetworkX graph
    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        nx_graph.add_node(sub_short, type='subject', original=subject)
        nx_graph.add_node(obj_short, type='object', original=obj)
        nx_graph.add_edge(sub_short, obj_short, label=pred_short, original_predicate=predicate)

    # Convert to undirected for connected components
    undirected_graph = nx_graph.to_undirected()
    components = list(nx.connected_components(undirected_graph))

    print(f"Found {len(components)} connected components")

    # Create subgraphs for each component
    component_subgraphs = []
    for i, component_nodes in enumerate(components):
        component_graph = rdflib.Graph()

        # Add all triples that involve nodes in this component
        for subject, predicate, obj in graph:
            sub_short = shorten_uri(str(subject))
            obj_short = shorten_uri(str(obj))

            if sub_short in component_nodes or obj_short in component_nodes:
                component_graph.add((subject, predicate, obj))

        component_subgraphs.append(component_graph)
        print(f"Component {i + 1}: {len(component_graph)} triples, {len(component_nodes)} nodes")

    return component_subgraphs, nx_graph, components


def create_clustered_visualization_high_quality(ttl_file, output_file="rdf_clustered.png", figsize=(20, 16), dpi=400):
    """
    Create high-quality clustered visualization - BEST QUALITY FOR COMPLEX GRAPHS
    """
    # Load the RDF graph
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')
    print(f"Loaded {len(graph)} triples")

    # Find connected components
    component_subgraphs, nx_graph, components = find_connected_components(graph)

    # Create the visualization with ultra high quality
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Use optimized layout for better component separation
    pos = nx.spring_layout(nx_graph, k=3, iterations=300, seed=42)

    # Enhanced color map for better distinction
    colors = plt.cm.tab20(np.linspace(0, 1, len(components)))

    # Calculate optimal sizes based on graph complexity
    total_nodes = len(nx_graph.nodes())
    if total_nodes > 100:
        node_size = 600
        font_size = 7
        edge_font_size = 5
    elif total_nodes > 50:
        node_size = 800
        font_size = 8
        edge_font_size = 6
    else:
        node_size = 1200
        font_size = 10
        edge_font_size = 8

    # Draw each component with distinct colors
    for i, component_nodes in enumerate(components):
        comp_subject_nodes = [node for node in component_nodes
                              if nx_graph.nodes[node].get('type') == 'subject']
        comp_object_nodes = [node for node in component_nodes
                             if nx_graph.nodes[node].get('type') == 'object']

        # Draw subject nodes
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=comp_subject_nodes,
                               node_color=[colors[i]], node_size=node_size,
                               alpha=0.9, edgecolors='black', linewidths=1.0)

        # Draw object nodes with slightly different shape indication
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=comp_object_nodes,
                               node_color=[colors[i]], node_size=node_size,
                               alpha=0.9, edgecolors='black', linewidths=1.0)

    # Draw edges with enhanced styling
    nx.draw_networkx_edges(nx_graph, pos, edge_color='#444444',
                           arrows=True, arrowsize=30, arrowstyle='->',
                           width=2.0, alpha=0.8)

    # Draw labels with improved readability
    nx.draw_networkx_labels(nx_graph, pos, font_size=font_size,
                            font_weight='bold', font_family='DejaVu Sans',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor='none'))

    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels,
                                 font_size=edge_font_size, font_family='DejaVu Sans',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.9, edgecolor='none'))

    plt.title(f"RDF Triples Visualization - {len(components)} Connected Components",
              fontsize=18, fontweight='bold', pad=25)
    plt.axis('off')
    plt.tight_layout()

    # Save with maximum quality (FIXED: removed optimize parameter)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                transparent=False, format='png')
    print(f" ULTRA-QUALITY Clustered visualization saved to: {os.path.abspath(output_file)}")
    plt.show()

    return component_subgraphs, nx_graph


def create_standard_visualization_high_quality(ttl_file, output_file="rdf_standard.png", figsize=(20, 16), dpi=400):
    """
    Create high-quality standard visualization - BEST FOR SIMPLE GRAPHS
    """
    # Load the RDF graph
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')
    print(f"Loaded {len(graph)} triples")

    # Create a NetworkX graph
    nx_graph = nx.DiGraph()

    # Add nodes and edges with shortened labels
    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        nx_graph.add_node(sub_short, type='subject')
        nx_graph.add_node(obj_short, type='object')
        nx_graph.add_edge(sub_short, obj_short, label=pred_short)

    # Create visualization with ultra high quality settings
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Use optimized layout
    try:
        pos = nx.kamada_kawai_layout(nx_graph)
    except:
        pos = nx.spring_layout(nx_graph, k=2, iterations=200, seed=42)

    # Calculate dynamic sizes based on graph complexity
    node_count = len(nx_graph.nodes())

    # Ultra-quality size adjustments
    if node_count > 100:
        node_size = 600
        font_size = 7
        edge_font_size = 5
    elif node_count > 50:
        node_size = 800
        font_size = 8
        edge_font_size = 6
    else:
        node_size = 1200
        font_size = 10
        edge_font_size = 8

    # Draw nodes with enhanced styling
    subject_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'subject']
    object_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'object']

    nx.draw_networkx_nodes(nx_graph, pos, nodelist=subject_nodes,
                           node_color='#1f77b4', node_size=node_size,
                           alpha=0.95, edgecolors='black', linewidths=1.0)
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=object_nodes,
                           node_color='#2ca02c', node_size=node_size,
                           alpha=0.95, edgecolors='black', linewidths=1.0)

    # Draw edges with enhanced styling
    nx.draw_networkx_edges(nx_graph, pos, edge_color='#444444',
                           arrows=True, arrowsize=30, arrowstyle='->',
                           width=2.0, alpha=0.8)

    # Draw node labels with improved readability
    nx.draw_networkx_labels(nx_graph, pos, font_size=font_size,
                            font_weight='bold', font_family='DejaVu Sans',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor='none'))

    # Draw edge labels with enhanced positioning
    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels,
                                 font_size=edge_font_size, font_family='DejaVu Sans',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.9, edgecolor='none'))

    plt.title("RDF Triples Visualization (Shortened URIs)",
              fontsize=18, fontweight='bold', pad=25)
    plt.axis('off')

    # Enhanced legend
    plt.scatter([], [], c='#1f77b4', label='Subjects', s=200, edgecolors='black')
    plt.scatter([], [], c='#2ca02c', label='Objects', s=200, edgecolors='black')
    plt.legend(scatterpoints=1, frameon=True, fancybox=True,
               shadow=True, framealpha=0.95, fontsize=14, loc='upper left')

    plt.tight_layout()

    # Save with maximum quality (FIXED: removed optimize parameter)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                transparent=False, format='png')
    print(f"ULTRA-QUALITY Standard visualization saved to: {os.path.abspath(output_file)}")

    plt.show()
    return nx_graph, graph


def choose_best_visualization(ttl_file, output_directory, dpi=400):
    """
    Automatically choose the best visualization method based on graph characteristics
    """
    # Load graph to analyze
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')

    # Analyze graph structure
    component_subgraphs, _, components = find_connected_components(graph)

    # Decision logic for best visualization
    if len(components) > 1:
        # Multiple connected components - use clustered visualization
        print("Multiple connected components detected - using CLUSTERED visualization")
        output_file = os.path.join(output_directory, "rdf_BEST_QUALITY_clustered.png")
        return create_clustered_visualization_high_quality(ttl_file, output_file, dpi=dpi)
    else:
        # Single connected component - use standard visualization
        print("Single connected component detected - using STANDARD visualization")
        output_file = os.path.join(output_directory, "rdf_BEST_QUALITY_standard.png")
        return create_standard_visualization_high_quality(ttl_file, output_file, dpi=dpi)


def export_ultra_quality_vector_formats(ttl_file, output_base_name="rdf_ultra_quality"):
    """
    Export to vector formats for maximum quality - BEST FOR PUBLICATIONS
    """
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')

    nx_graph = nx.DiGraph()
    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))
        nx_graph.add_node(sub_short, type='subject')
        nx_graph.add_node(obj_short, type='object')
        nx_graph.add_edge(sub_short, obj_short, label=pred_short)

    pos = nx.spring_layout(nx_graph, k=2, iterations=200, seed=42)

    # Create ultra-quality plot
    plt.figure(figsize=(20, 16))

    # Enhanced graph elements
    subject_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'subject']
    object_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'object']

    nx.draw_networkx_nodes(nx_graph, pos, nodelist=subject_nodes,
                           node_color='#1f77b4', node_size=1200, alpha=0.95,
                           edgecolors='black', linewidths=1.0)
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=object_nodes,
                           node_color='#2ca02c', node_size=1200, alpha=0.95,
                           edgecolors='black', linewidths=1.0)
    nx.draw_networkx_edges(nx_graph, pos, edge_color='#444444',
                           arrows=True, arrowsize=25, arrowstyle='->',
                           width=2.0, alpha=0.8)
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor='none'))

    edge_labels = {(u, v): d['label'] for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels,
                                 font_size=8,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.9, edgecolor='none'))

    plt.title("RDF Triples Visualization - Ultra Quality", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Ultra-quality exports (FIXED: removed optimize parameter)
    plt.savefig(f"{output_base_name}.svg", format='svg', bbox_inches='tight')
    print(f"ULTRA-QUALITY SVG saved to: {os.path.abspath(output_base_name)}.svg")

    plt.savefig(f"{output_base_name}.pdf", format='pdf', bbox_inches='tight')
    print(f"ULTRA-QUALITY PDF saved to: {os.path.abspath(output_base_name)}.pdf")

    plt.savefig(f"{output_base_name}_ultra.png", dpi=600, bbox_inches='tight')
    print(f"ULTRA-QUALITY PNG saved to: {os.path.abspath(output_base_name)}_ultra.png")

    plt.close()


def print_triples_table(graph, max_triples=20):
    """
    Print triples in a table format with shortened URIs
    """
    print("\n" + "=" * 80)
    print("RDF TRIPLES (Shortened URIs)")
    print("=" * 80)
    print(f"{'Subject':<30} {'Predicate':<25} {'Object':<30}")
    print("-" * 80)

    count = 0
    for subject, predicate, obj in graph:
        if count >= max_triples:
            print(f"... and {len(graph) - max_triples} more triples")
            break

        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        print(f"{sub_short:<30} {pred_short:<25} {obj_short:<30}")
        count += 1


def create_interactive_visualization(ttl_file, output_html="rdf_interactive.html"):
    """
    Create an interactive visualization using pyvis
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("Pyvis not installed. Install with: pip install pyvis")
        return

    # Load the RDF graph
    graph = rdflib.Graph()
    graph.parse(ttl_file, format='turtle')

    # Create pyvis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Add nodes and edges
    node_ids = {}
    node_counter = 0

    for subject, predicate, obj in graph:
        sub_short = shorten_uri(str(subject))
        pred_short = shorten_uri(str(predicate))
        obj_short = shorten_uri(str(obj))

        # Add subject node
        if sub_short not in node_ids:
            node_ids[sub_short] = node_counter
            net.add_node(node_counter, label=sub_short, color='#97c2fc', shape='ellipse')
            node_counter += 1

        # Add object node
        if obj_short not in node_ids:
            node_ids[obj_short] = node_counter
            net.add_node(node_counter, label=obj_short, color='#98FB98', shape='box')
            node_counter += 1

        # Add edge
        net.add_edge(node_ids[sub_short], node_ids[obj_short],
                     title=pred_short, label=pred_short, color='white')

    # Configure physics for better layout
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)

    net.show(output_html)
    print(f"Interactive visualization saved to: {os.path.abspath(output_html)}")


# Quick filter functions for common use cases
def filter_by_type(graph, type_name=None):
    """Filter triples by type predicate, optionally by specific type"""
    if type_name:
        return filter_triples(graph, predicate_filter='type', object_filter=type_name)
    else:
        return filter_triples(graph, predicate_filter='type')


def filter_by_label(graph):
    """Filter triples by label predicate"""
    return filter_triples(graph, predicate_filter='label')


# MAIN EXECUTION - AUTOMATIC BEST QUALITY SELECTION
if __name__ == "__main__":
    # Your TTL file path
    ttl_file_path = "/Users/malikluti/Desktop/UCI/Code/cornercase/output/bus_obscuring_car_second_car/day/6/merged_observation_loop.ttl"

    # Output directory
    output_directory = "/Users/malikluti/Desktop/UCI/Visualizations/"
    os.makedirs(output_directory, exist_ok=True)

    try:
        print(" Analyzing RDF graph structure for optimal visualization...")

        # OPTION 1: AUTOMATIC BEST QUALITY VISUALIZATION (Recommended)
        print("\n Creating AUTOMATIC BEST QUALITY visualization...")
        best_result = choose_best_visualization(
            ttl_file_path,
            output_directory,
            dpi=400  # Ultra high DPI
        )

        # Print triples table
        if isinstance(best_result[0], list):  # Clustered result
            print_triples_table(best_result[0][0])  # Print first component
        else:  # Standard result
            print_triples_table(best_result[1])

        # OPTION 2: ULTRA-QUALITY VECTOR FORMATS (Best for publications)
        print("\n  Exporting ULTRA-QUALITY vector formats...")
        vector_base = os.path.join(output_directory, "rdf_ultra_quality")
        export_ultra_quality_vector_formats(ttl_file_path, vector_base)

        # OPTION 3: Interactive visualization
        print("\n Creating interactive visualization...")
        interactive_file = os.path.join(output_directory, "rdf_interactive.html")
        create_interactive_visualization(ttl_file_path, interactive_file)

        print("\n ALL ULTRA-QUALITY VISUALIZATIONS COMPLETED!")
        print(f" Files saved to: {output_directory}")

        print("\n GENERATED FILES:")
        print(f"  • rdf_BEST_QUALITY_*.png (Automatically chosen best visualization)")
        print(f"  • rdf_ultra_quality.svg (Publication-quality vector)")
        print(f"  • rdf_ultra_quality.pdf (Publication-quality PDF)")
        print(f"  • rdf_ultra_quality_ultra.png (Ultra high-res PNG)")
        print(f"  • rdf_interactive.html (Interactive web version)")

    except FileNotFoundError:
        print(f" File {ttl_file_path} not found. Please check the file path.")
    except Exception as e:
        print(f" Error: {e}")