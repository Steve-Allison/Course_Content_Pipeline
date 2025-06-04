"""
Knowledge Graph Builder

Creates semantic knowledge graphs from extracted content to map relationships
between concepts, entities, and content pieces for enhanced AI reuse.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import networkx as nx

# NLP and ML libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.offline as pyo
    GRAPH_VIZ_AVAILABLE = True
except ImportError:
    GRAPH_VIZ_AVAILABLE = False

from .logging_utils import get_logger

logger = get_logger(__name__)

class KnowledgeGraphBuilder:
    """Builds comprehensive knowledge graphs from extracted content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'output/knowledge_graph')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.min_concept_frequency = self.config.get('min_concept_frequency', 2)

        # Initialize models
        self.nlp = None
        self.sentence_model = None
        self._load_models()

        # Knowledge graph
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}
        self.entity_relationships = defaultdict(list)

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_models(self):
        """Load NLP models for knowledge extraction."""
        try:
            if SPACY_AVAILABLE:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for NLP processing")

            if EMBEDDINGS_AVAILABLE:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer for embeddings")

        except Exception as e:
            logger.warning(f"Failed to load NLP models: {e}")

    def build_knowledge_graph(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive knowledge graph from all extracted content.

        Args:
            content_data: Dictionary containing all extracted content

        Returns:
            Knowledge graph analysis and statistics
        """
        logger.info("Building knowledge graph from extracted content")

        try:
            # 1. Extract concepts and entities
            concepts_entities = self._extract_concepts_entities(content_data)

            # 2. Build semantic relationships
            relationships = self._build_semantic_relationships(concepts_entities, content_data)

            # 3. Create cross-document connections
            cross_connections = self._create_cross_document_connections(content_data)

            # 4. Build hierarchical structures
            hierarchies = self._build_content_hierarchies(content_data)

            # 5. Generate embeddings for semantic search
            embeddings_map = self._generate_concept_embeddings(concepts_entities)

            # 6. Construct the graph
            self._construct_graph(concepts_entities, relationships, cross_connections, hierarchies)

            # 7. Analyze graph properties
            graph_analysis = self._analyze_graph_properties()

            # 8. Generate AI-ready insights
            ai_insights = self._generate_ai_insights(graph_analysis)

            # 9. Save graph data
            graph_data = self._export_graph_data()

            return {
                'concepts_entities': concepts_entities,
                'relationships': relationships,
                'cross_connections': cross_connections,
                'hierarchies': hierarchies,
                'embeddings_map': embeddings_map,
                'graph_analysis': graph_analysis,
                'ai_insights': ai_insights,
                'graph_data': graph_data,
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges()
            }

        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}", exc_info=True)
            return {'error': str(e)}

    def _extract_concepts_entities(self, content_data: Dict) -> Dict[str, Any]:
        """Extract concepts and entities from all content types."""
        concepts = defaultdict(list)
        entities = defaultdict(list)

        # Process instructional content
        instructional_assets = content_data.get('instructional_assets', [])
        for asset in instructional_assets:
            if isinstance(asset, dict) and 'segments' in asset:
                for segment in asset['segments']:
                    source_concepts, source_entities = self._process_segment(segment)
                    concepts['instructional'].extend(source_concepts)
                    entities['instructional'].extend(source_entities)

        # Process caption segments
        caption_segments = content_data.get('caption_segments', [])
        for segment in caption_segments:
            source_concepts, source_entities = self._process_segment(segment)
            concepts['captions'].extend(source_concepts)
            entities['captions'].extend(source_entities)

        # Process audio analysis if available
        if 'audio_analysis' in content_data:
            audio_concepts = self._extract_audio_concepts(content_data['audio_analysis'])
            concepts['audio'].extend(audio_concepts)

        # Process visual analysis if available
        if 'visual_analysis' in content_data:
            visual_concepts = self._extract_visual_concepts(content_data['visual_analysis'])
            concepts['visual'].extend(visual_concepts)

        # Consolidate and rank concepts
        all_concepts = []
        for source, concept_list in concepts.items():
            all_concepts.extend([(concept, source) for concept in concept_list])

        concept_frequency = Counter([concept for concept, _ in all_concepts])

        # Filter by frequency
        filtered_concepts = {
            concept: {
                'frequency': freq,
                'sources': [source for c, source in all_concepts if c == concept]
            }
            for concept, freq in concept_frequency.items()
            if freq >= self.min_concept_frequency
        }

        return {
            'concepts': filtered_concepts,
            'entities': dict(entities),
            'concept_frequency': dict(concept_frequency),
            'total_concepts': len(filtered_concepts),
            'total_entities': sum(len(entity_list) for entity_list in entities.values())
        }

    def _process_segment(self, segment: Dict) -> Tuple[List[str], List[str]]:
        """Process a content segment to extract concepts and entities."""
        concepts = []
        entities = []

        try:
            text = segment.get('text', '')
            if not text:
                return concepts, entities

            # Extract from existing tags/topics if available
            if 'topics' in segment:
                concepts.extend(segment['topics'])

            if 'tags' in segment:
                concepts.extend(segment['tags'])

            # Extract entities from existing analysis
            if 'entities' in segment:
                entities.extend(segment['entities'])

            # Use spaCy for additional extraction if available
            if self.nlp and SPACY_AVAILABLE:
                doc = self.nlp(text)

                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        })

                # Extract noun phrases as potential concepts
                for np in doc.noun_chunks:
                    if len(np.text.split()) <= 3 and len(np.text) > 3:  # Filter short phrases
                        concepts.append(np.text.lower().strip())

                # Extract key terms (nouns and proper nouns)
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and
                        not token.is_stop and
                        not token.is_punct and
                        len(token.text) > 2):
                        concepts.append(token.lemma_.lower())

            return list(set(concepts)), entities

        except Exception as e:
            logger.warning(f"Error processing segment: {e}")
            return concepts, entities

    def _build_semantic_relationships(self, concepts_entities: Dict, content_data: Dict) -> List[Dict]:
        """Build semantic relationships between concepts."""
        relationships = []

        try:
            concepts = concepts_entities.get('concepts', {})
            concept_list = list(concepts.keys())

            if EMBEDDINGS_AVAILABLE and self.sentence_model:
                # Generate embeddings for concepts
                concept_embeddings = self.sentence_model.encode(concept_list)

                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(concept_embeddings)

                # Find related concepts
                for i, concept1 in enumerate(concept_list):
                    for j, concept2 in enumerate(concept_list):
                        if i != j and similarity_matrix[i][j] > self.similarity_threshold:
                            relationships.append({
                                'source': concept1,
                                'target': concept2,
                                'relationship_type': 'semantic_similarity',
                                'strength': float(similarity_matrix[i][j]),
                                'evidence': 'embedding_similarity'
                            })

            # Add co-occurrence relationships
            cooccurrence_rels = self._find_cooccurrence_relationships(concepts_entities, content_data)
            relationships.extend(cooccurrence_rels)

            # Add hierarchical relationships
            hierarchical_rels = self._find_hierarchical_relationships(concepts_entities)
            relationships.extend(hierarchical_rels)

            return relationships

        except Exception as e:
            logger.warning(f"Error building semantic relationships: {e}")
            return relationships

    def _find_cooccurrence_relationships(self, concepts_entities: Dict, content_data: Dict) -> List[Dict]:
        """Find concepts that frequently co-occur in the same content."""
        relationships = []
        concept_cooccurrence = defaultdict(lambda: defaultdict(int))

        # Track co-occurrence in segments
        all_segments = []

        # Collect all segments
        instructional_assets = content_data.get('instructional_assets', [])
        for asset in instructional_assets:
            if isinstance(asset, dict) and 'segments' in asset:
                all_segments.extend(asset['segments'])

        caption_segments = content_data.get('caption_segments', [])
        all_segments.extend(caption_segments)

        # Find co-occurring concepts
        for segment in all_segments:
            segment_text = segment.get('text', '').lower()
            segment_concepts = []

            for concept in concepts_entities.get('concepts', {}):
                if concept.lower() in segment_text:
                    segment_concepts.append(concept)

            # Record co-occurrences
            for i, concept1 in enumerate(segment_concepts):
                for concept2 in segment_concepts[i+1:]:
                    concept_cooccurrence[concept1][concept2] += 1
                    concept_cooccurrence[concept2][concept1] += 1

        # Create relationships for frequent co-occurrences
        min_cooccurrence = 2
        for concept1, related_concepts in concept_cooccurrence.items():
            for concept2, count in related_concepts.items():
                if count >= min_cooccurrence:
                    relationships.append({
                        'source': concept1,
                        'target': concept2,
                        'relationship_type': 'co_occurrence',
                        'strength': min(count / 10.0, 1.0),  # Normalize strength
                        'evidence': f'co_occurred_{count}_times'
                    })

        return relationships

    def _find_hierarchical_relationships(self, concepts_entities: Dict) -> List[Dict]:
        """Find hierarchical relationships between concepts."""
        relationships = []
        concepts = list(concepts_entities.get('concepts', {}).keys())

        # Simple heuristic: shorter concepts might be parents of longer ones
        for concept1 in concepts:
            for concept2 in concepts:
                if concept1 != concept2:
                    # Check if concept1 is contained in concept2
                    if concept1.lower() in concept2.lower() and len(concept1) < len(concept2):
                        relationships.append({
                            'source': concept1,
                            'target': concept2,
                            'relationship_type': 'hierarchical_parent',
                            'strength': 0.8,
                            'evidence': 'substring_containment'
                        })

        return relationships

    def _create_cross_document_connections(self, content_data: Dict) -> List[Dict]:
        """Create connections between content across different documents."""
        connections = []

        try:
            # Find similar content across documents
            if EMBEDDINGS_AVAILABLE and self.sentence_model:
                documents = []
                doc_metadata = []

                # Collect document summaries
                instructional_assets = content_data.get('instructional_assets', [])
                for i, asset in enumerate(instructional_assets):
                    if isinstance(asset, dict):
                        # Use title, summary, or first segment as document representation
                        doc_text = (asset.get('title', '') + ' ' +
                                   asset.get('summary', '') + ' ' +
                                   str(asset.get('segments', [{}])[0].get('text', '') if asset.get('segments') else ''))
                        documents.append(doc_text)
                        doc_metadata.append({
                            'type': 'instructional',
                            'index': i,
                            'title': asset.get('title', f'Document_{i}')
                        })

                # Generate embeddings and find similarities
                if documents:
                    embeddings = self.sentence_model.encode(documents)
                    similarity_matrix = cosine_similarity(embeddings)

                    for i in range(len(documents)):
                        for j in range(i+1, len(documents)):
                            if similarity_matrix[i][j] > 0.6:  # Threshold for document similarity
                                connections.append({
                                    'source_doc': doc_metadata[i],
                                    'target_doc': doc_metadata[j],
                                    'connection_type': 'content_similarity',
                                    'strength': float(similarity_matrix[i][j]),
                                    'evidence': 'document_embedding_similarity'
                                })

            return connections

        except Exception as e:
            logger.warning(f"Error creating cross-document connections: {e}")
            return connections

    def _build_content_hierarchies(self, content_data: Dict) -> Dict[str, Any]:
        """Build hierarchical structures from content organization."""
        hierarchies = {
            'instructional_hierarchy': [],
            'topic_hierarchy': {},
            'source_hierarchy': {}
        }

        try:
            # Build instructional content hierarchy
            instructional_assets = content_data.get('instructional_assets', [])
            for asset in instructional_assets:
                if isinstance(asset, dict):
                    hierarchy_node = {
                        'title': asset.get('title', 'Untitled'),
                        'type': asset.get('type', 'unknown'),
                        'children': []
                    }

                    # Add segments as children
                    segments = asset.get('segments', [])
                    for segment in segments:
                        if isinstance(segment, dict):
                            segment_node = {
                                'text': segment.get('text', '')[:100] + '...' if len(segment.get('text', '')) > 100 else segment.get('text', ''),
                                'topics': segment.get('topics', []),
                                'tags': segment.get('tags', [])
                            }
                            hierarchy_node['children'].append(segment_node)

                    hierarchies['instructional_hierarchy'].append(hierarchy_node)

            # Build topic hierarchy
            all_topics = []
            caption_segments = content_data.get('caption_segments', [])
            for segment in caption_segments:
                if isinstance(segment, dict) and 'topics' in segment:
                    all_topics.extend(segment['topics'])

            topic_freq = Counter(all_topics)
            hierarchies['topic_hierarchy'] = dict(topic_freq)

            return hierarchies

        except Exception as e:
            logger.warning(f"Error building content hierarchies: {e}")
            return hierarchies

    def _generate_concept_embeddings(self, concepts_entities: Dict) -> Dict[str, Any]:
        """Generate embeddings for all concepts for semantic search."""
        embeddings_map = {}

        try:
            if EMBEDDINGS_AVAILABLE and self.sentence_model:
                concepts = list(concepts_entities.get('concepts', {}).keys())

                if concepts:
                    embeddings = self.sentence_model.encode(concepts)

                    for i, concept in enumerate(concepts):
                        embeddings_map[concept] = {
                            'embedding': embeddings[i].tolist(),
                            'dimension': len(embeddings[i])
                        }

                    logger.info(f"Generated embeddings for {len(concepts)} concepts")

            return embeddings_map

        except Exception as e:
            logger.warning(f"Error generating concept embeddings: {e}")
            return embeddings_map

    def _construct_graph(self, concepts_entities: Dict, relationships: List[Dict],
                        cross_connections: List[Dict], hierarchies: Dict):
        """Construct the NetworkX graph from extracted data."""
        try:
            # Add concept nodes
            concepts = concepts_entities.get('concepts', {})
            for concept, data in concepts.items():
                self.graph.add_node(concept,
                                  node_type='concept',
                                  frequency=data['frequency'],
                                  sources=data['sources'])

            # Add entity nodes
            entities = concepts_entities.get('entities', {})
            for source, entity_list in entities.items():
                for entity in entity_list:
                    if isinstance(entity, dict):
                        entity_text = entity.get('text', '')
                        self.graph.add_node(entity_text,
                                          node_type='entity',
                                          label=entity.get('label', ''),
                                          source=source)

            # Add relationship edges
            for rel in relationships:
                self.graph.add_edge(rel['source'], rel['target'],
                                  relationship_type=rel['relationship_type'],
                                  strength=rel['strength'],
                                  evidence=rel['evidence'])

            # Add cross-document connections as edges
            for conn in cross_connections:
                source_title = conn['source_doc']['title']
                target_title = conn['target_doc']['title']

                # Add document nodes if not present
                if not self.graph.has_node(source_title):
                    self.graph.add_node(source_title, node_type='document')
                if not self.graph.has_node(target_title):
                    self.graph.add_node(target_title, node_type='document')

                self.graph.add_edge(source_title, target_title,
                                  relationship_type=conn['connection_type'],
                                  strength=conn['strength'])

            logger.info(f"Constructed graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

        except Exception as e:
            logger.error(f"Error constructing graph: {e}")

    def _analyze_graph_properties(self) -> Dict[str, Any]:
        """Analyze properties of the constructed knowledge graph."""
        analysis = {}

        try:
            # Basic statistics
            analysis['basic_stats'] = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph)
            }

            # Node analysis
            node_types = defaultdict(int)
            for node, data in self.graph.nodes(data=True):
                node_types[data.get('node_type', 'unknown')] += 1
            analysis['node_types'] = dict(node_types)

            # Centrality measures
            if self.graph.number_of_nodes() > 0:
                try:
                    degree_centrality = nx.degree_centrality(self.graph)
                    betweenness_centrality = nx.betweenness_centrality(self.graph)

                    # Top central concepts
                    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

                    analysis['centrality'] = {
                        'top_degree_centrality': top_degree,
                        'top_betweenness_centrality': top_betweenness
                    }
                except:
                    analysis['centrality'] = {'error': 'Could not calculate centrality measures'}

            # Community detection
            if self.graph.number_of_nodes() > 2:
                try:
                    # Convert to undirected for community detection
                    undirected_graph = self.graph.to_undirected()
                    communities = nx.community.greedy_modularity_communities(undirected_graph)

                    analysis['communities'] = {
                        'num_communities': len(communities),
                        'community_sizes': [len(c) for c in communities],
                        'modularity': nx.community.modularity(undirected_graph, communities)
                    }
                except:
                    analysis['communities'] = {'error': 'Could not detect communities'}

            return analysis

        except Exception as e:
            logger.warning(f"Error analyzing graph properties: {e}")
            return {'error': str(e)}

    def _generate_ai_insights(self, graph_analysis: Dict) -> Dict[str, Any]:
        """Generate AI-ready insights from the knowledge graph."""
        insights = {
            'key_concepts': [],
            'concept_clusters': [],
            'learning_paths': [],
            'knowledge_gaps': [],
            'ai_prompts': []
        }

        try:
            # Identify key concepts from centrality
            centrality = graph_analysis.get('centrality', {})
            if 'top_degree_centrality' in centrality:
                insights['key_concepts'] = [
                    {
                        'concept': concept,
                        'importance_score': score,
                        'reasoning': 'High degree centrality in knowledge graph'
                    }
                    for concept, score in centrality['top_degree_centrality'][:5]
                ]

            # Generate concept clusters
            communities = graph_analysis.get('communities', {})
            if 'num_communities' in communities:
                insights['concept_clusters'] = [
                    {
                        'cluster_id': i,
                        'size': size,
                        'learning_objective': f"Understand relationships in concept cluster {i+1}"
                    }
                    for i, size in enumerate(communities.get('community_sizes', []))
                ]

            # Suggest learning paths based on graph structure
            if self.graph.number_of_nodes() > 0:
                try:
                    # Find concepts with high betweenness centrality as pathway nodes
                    betweenness = graph_analysis.get('centrality', {}).get('top_betweenness_centrality', [])
                    if betweenness:
                        pathway_concepts = [concept for concept, _ in betweenness[:3]]
                        insights['learning_paths'] = [
                            {
                                'path_type': 'concept_pathway',
                                'key_concepts': pathway_concepts,
                                'description': 'Learning path through central concepts'
                            }
                        ]
                except:
                    pass

            # Identify potential knowledge gaps
            isolated_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
            if isolated_nodes:
                insights['knowledge_gaps'] = [
                    {
                        'type': 'isolated_concepts',
                        'concepts': isolated_nodes[:10],  # Limit to top 10
                        'recommendation': 'Create connections between isolated concepts and main knowledge network'
                    }
                ]

            # Generate AI prompts for content enhancement
            key_concepts = [item['concept'] for item in insights['key_concepts'][:3]]
            if key_concepts:
                insights['ai_prompts'] = [
                    f"Create comprehensive learning materials that connect these key concepts: {', '.join(key_concepts)}",
                    f"Design assessment questions that test understanding of relationships between {key_concepts[0]} and {key_concepts[1] if len(key_concepts) > 1 else 'related concepts'}",
                    f"Develop practical examples demonstrating the application of {key_concepts[0]} in real-world scenarios"
                ]

            return insights

        except Exception as e:
            logger.warning(f"Error generating AI insights: {e}")
            return insights

    def _export_graph_data(self) -> Dict[str, Any]:
        """Export graph data in multiple formats for AI consumption."""
        export_data = {}

        try:
            # Export as adjacency list
            export_data['adjacency_list'] = dict(self.graph.adjacency())

            # Export nodes with attributes
            export_data['nodes'] = [
                {'id': node, **data}
                for node, data in self.graph.nodes(data=True)
            ]

            # Export edges with attributes
            export_data['edges'] = [
                {'source': source, 'target': target, **data}
                for source, target, data in self.graph.edges(data=True)
            ]

            # Export in format suitable for visualization
            if GRAPH_VIZ_AVAILABLE:
                export_data['visualization_ready'] = {
                    'nodes': [
                        {
                            'id': node,
                            'label': node,
                            'type': data.get('node_type', 'unknown'),
                            'size': data.get('frequency', 1) * 10
                        }
                        for node, data in self.graph.nodes(data=True)
                    ],
                    'links': [
                        {
                            'source': source,
                            'target': target,
                            'strength': data.get('strength', 0.5),
                            'type': data.get('relationship_type', 'unknown')
                        }
                        for source, target, data in self.graph.edges(data=True)
                    ]
                }

            # Save graph files
            self._save_graph_files(export_data)

            return export_data

        except Exception as e:
            logger.warning(f"Error exporting graph data: {e}")
            return {'error': str(e)}

    def _save_graph_files(self, export_data: Dict):
        """Save graph in various formats."""
        try:
            # Save as JSON
            graph_json_path = Path(self.output_dir) / 'knowledge_graph.json'
            with open(graph_json_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            # Save as GraphML for external tools
            if self.graph.number_of_nodes() > 0:
                graphml_path = Path(self.output_dir) / 'knowledge_graph.graphml'
                nx.write_graphml(self.graph, graphml_path)

            # Save embeddings separately
            if self.concept_embeddings:
                embeddings_path = Path(self.output_dir) / 'concept_embeddings.json'
                with open(embeddings_path, 'w') as f:
                    json.dump(self.concept_embeddings, f, indent=2)

            logger.info(f"Saved knowledge graph files to {self.output_dir}")

        except Exception as e:
            logger.warning(f"Error saving graph files: {e}")

    def _extract_audio_concepts(self, audio_analysis: Dict) -> List[str]:
        """Extract concepts from audio analysis results."""
        concepts = []

        # Extract from transcription topics/tags if available
        for file_analysis in audio_analysis.values():
            if isinstance(file_analysis, dict):
                transcription = file_analysis.get('transcription', {})
                if 'text' in transcription:
                    # Simple keyword extraction from transcription
                    text = transcription['text'].lower()
                    # This could be enhanced with more sophisticated audio concept extraction
                    concepts.extend(text.split()[:10])  # Simple approach

        return concepts

    def _extract_visual_concepts(self, visual_analysis: Dict) -> List[str]:
        """Extract concepts from visual analysis results."""
        concepts = []

        # Extract from OCR and AI descriptions
        for image_analysis in visual_analysis.values():
            if isinstance(image_analysis, dict):
                # From OCR text
                ocr_analysis = image_analysis.get('ocr_analysis', {})
                if 'full_text' in ocr_analysis:
                    concepts.extend(ocr_analysis['full_text'].split()[:5])

                # From AI descriptions
                ai_description = image_analysis.get('ai_description', {})
                if 'general_caption' in ai_description:
                    concepts.extend(ai_description['general_caption'].split()[:5])

                # From knowledge extraction
                knowledge_extraction = image_analysis.get('knowledge_extraction', {})
                if 'key_concepts' in knowledge_extraction:
                    concepts.extend(knowledge_extraction['key_concepts'])

        return concepts

# Convenience function for integration
def build_comprehensive_knowledge_graph(content_data: Dict, output_dir: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a comprehensive knowledge graph from all extracted content."""
    builder = KnowledgeGraphBuilder(config)
    builder.output_dir = output_dir

    return builder.build_knowledge_graph(content_data)
