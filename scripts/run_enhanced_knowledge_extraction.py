"""
Enhanced Knowledge Extraction Runner

Integrates advanced visual content analysis and knowledge graph building
to maximize knowledge extraction for AI downstream processing.
"""

import logging
import datetime
import json
from pathlib import Path
from course_compiler.config import (
    ANALYSIS_DIR, LOGS_DIR, PROCESSED_DIR, INSTRUCTIONAL_JSON_DIR
)
from course_compiler.visual_content_analysis import analyze_visual_content_directory
from course_compiler.knowledge_graph_builder import build_comprehensive_knowledge_graph

# Set up logging
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
logfile = Path(LOGS_DIR) / f"enhanced_knowledge_extraction_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    print("=== Enhanced Knowledge Extraction ===")
    logging.info("Starting enhanced knowledge extraction...")

    # Create analysis output directories
    visual_analysis_dir = Path(ANALYSIS_DIR) / "visual_content"
    knowledge_graph_dir = Path(ANALYSIS_DIR) / "knowledge_graph"
    enhanced_insights_dir = Path(ANALYSIS_DIR) / "enhanced_insights"

    visual_analysis_dir.mkdir(parents=True, exist_ok=True)
    knowledge_graph_dir.mkdir(parents=True, exist_ok=True)
    enhanced_insights_dir.mkdir(parents=True, exist_ok=True)

    # Configuration for enhanced analysis
    visual_config = {
        'output_dir': str(visual_analysis_dir),
        'include_ai_analysis': True,
        'include_ocr': True,
        'include_chart_analysis': True
    }

    knowledge_graph_config = {
        'output_dir': str(knowledge_graph_dir),
        'similarity_threshold': 0.7,
        'min_concept_frequency': 2,
        'enable_visualization': True
    }

    total_insights = {}

    # Phase 1: Visual Content Analysis
    print("\n🔍 Phase 1: Visual Content Analysis")
    logging.info("Starting visual content analysis...")

    try:
        # Analyze images from extracted content
        images_dir = Path(INSTRUCTIONAL_JSON_DIR) / "images"
        if images_dir.exists():
            print(f"Analyzing visual content in: {images_dir}")

            visual_results = analyze_visual_content_directory(
                input_dir=str(images_dir),
                output_dir=str(visual_analysis_dir),
                config=visual_config
            )

            total_insights['visual_analysis'] = visual_results
            print(f"✅ Analyzed {len(visual_results)} visual assets")
            logging.info(f"Visual analysis complete: {len(visual_results)} files")

            # Generate visual insights summary
            visual_summary = generate_visual_insights_summary(visual_results)

            summary_path = visual_analysis_dir / "visual_insights_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(visual_summary, f, indent=2, default=str)

            print(f"📊 Visual insights summary saved to: {summary_path}")
        else:
            print(f"⚠️ Images directory not found: {images_dir}")
            total_insights['visual_analysis'] = {}

    except Exception as e:
        print(f"❌ Error in visual content analysis: {e}")
        logging.error(f"Visual content analysis failed: {e}", exc_info=True)
        total_insights['visual_analysis'] = {'error': str(e)}

    # Phase 2: Load Existing Content Data
    print("\n📚 Phase 2: Loading Content Data")
    logging.info("Loading existing content data...")

    try:
        content_data = load_all_content_data()

        # Integrate visual analysis results
        if 'visual_analysis' in total_insights:
            content_data['visual_analysis'] = total_insights['visual_analysis']

        print(f"✅ Loaded content data with {len(content_data)} content types")
        logging.info(f"Content data loaded successfully")

    except Exception as e:
        print(f"❌ Error loading content data: {e}")
        logging.error(f"Failed to load content data: {e}", exc_info=True)
        content_data = {}

    # Phase 3: Knowledge Graph Construction
    print("\n🧠 Phase 3: Knowledge Graph Construction")
    logging.info("Building comprehensive knowledge graph...")

    try:
        if content_data:
            print("Building knowledge graph from all extracted content...")

            knowledge_graph_results = build_comprehensive_knowledge_graph(
                content_data=content_data,
                output_dir=str(knowledge_graph_dir),
                config=knowledge_graph_config
            )

            total_insights['knowledge_graph'] = knowledge_graph_results

            if 'total_nodes' in knowledge_graph_results:
                print(f"✅ Knowledge graph built: {knowledge_graph_results['total_nodes']} nodes, {knowledge_graph_results['total_edges']} edges")
                logging.info(f"Knowledge graph complete: {knowledge_graph_results['total_nodes']} nodes, {knowledge_graph_results['total_edges']} edges")
            else:
                print("✅ Knowledge graph analysis completed")
                logging.info("Knowledge graph analysis completed")

        else:
            print("⚠️ No content data available for knowledge graph construction")
            total_insights['knowledge_graph'] = {'error': 'No content data available'}

    except Exception as e:
        print(f"❌ Error in knowledge graph construction: {e}")
        logging.error(f"Knowledge graph construction failed: {e}", exc_info=True)
        total_insights['knowledge_graph'] = {'error': str(e)}

    # Phase 4: Generate Enhanced AI Insights
    print("\n🎯 Phase 4: Enhanced AI Insights Generation")
    logging.info("Generating enhanced AI insights...")

    try:
        enhanced_insights = generate_enhanced_ai_insights(total_insights)

        # Save comprehensive insights
        insights_path = enhanced_insights_dir / "enhanced_ai_insights.json"
        with open(insights_path, 'w') as f:
            json.dump(enhanced_insights, f, indent=2, default=str)

        # Generate AI-ready prompt bundle
        prompt_bundle = generate_ai_prompt_bundle(enhanced_insights)

        prompt_bundle_path = enhanced_insights_dir / "enhanced_ai_prompts.md"
        with open(prompt_bundle_path, 'w') as f:
            f.write(prompt_bundle)

        print(f"✅ Enhanced insights generated")
        print(f"📋 AI insights: {insights_path}")
        print(f"🤖 AI prompts: {prompt_bundle_path}")
        logging.info("Enhanced AI insights generation complete")

    except Exception as e:
        print(f"❌ Error generating enhanced insights: {e}")
        logging.error(f"Enhanced insights generation failed: {e}", exc_info=True)

    # Summary
    print(f"\n=== Enhanced Knowledge Extraction Complete ===")
    print(f"📁 Visual analysis: {visual_analysis_dir}")
    print(f"🧠 Knowledge graph: {knowledge_graph_dir}")
    print(f"🎯 Enhanced insights: {enhanced_insights_dir}")
    print(f"📋 Detailed logs: {logfile}")

    # Generate final summary
    generate_final_summary(total_insights, enhanced_insights_dir)

    logging.info("Enhanced knowledge extraction complete")

def load_all_content_data():
    """Load all existing content data for knowledge graph construction."""
    content_data = {}

    try:
        # Load master summary if available
        master_summary_path = Path(PROCESSED_DIR).parent / "summary" / "master_summary.json"
        if master_summary_path.exists():
            with open(master_summary_path, 'r') as f:
                content_data = json.load(f)
            logging.info(f"Loaded master summary from {master_summary_path}")

        # Load additional analysis data
        analysis_files = [
            "entities_summary.json",
            "topics_summary.json",
            "audio_features/audio_analysis_summary.json"
        ]

        for analysis_file in analysis_files:
            file_path = Path(ANALYSIS_DIR).parent / "summary" / analysis_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Add to content data with appropriate key
                    key = analysis_file.split('.')[0].replace('_summary', '')
                    content_data[key] = data
                    logging.info(f"Loaded {analysis_file}")
                except Exception as e:
                    logging.warning(f"Failed to load {analysis_file}: {e}")

        return content_data

    except Exception as e:
        logging.error(f"Error loading content data: {e}")
        return {}

def generate_visual_insights_summary(visual_results):
    """Generate a summary of visual analysis insights."""
    summary = {
        'analysis_timestamp': str(datetime.datetime.now()),
        'total_images_analyzed': len(visual_results),
        'content_categories': {},
        'key_concepts_extracted': [],
        'educational_classifications': {},
        'ai_actionable_insights': [],
        'ocr_text_extracted': 0,
        'charts_detected': 0
    }

    try:
        all_concepts = []
        for file_path, analysis in visual_results.items():
            if isinstance(analysis, dict) and 'error' not in analysis:
                # Content categories
                edu_class = analysis.get('educational_classification', {})
                category = edu_class.get('content_category', 'unknown')
                summary['content_categories'][category] = summary['content_categories'].get(category, 0) + 1

                # Key concepts
                knowledge_ext = analysis.get('knowledge_extraction', {})
                concepts = knowledge_ext.get('key_concepts', [])
                all_concepts.extend(concepts)

                # OCR text
                ocr_analysis = analysis.get('ocr_analysis', {})
                if 'full_text' in ocr_analysis and ocr_analysis['full_text']:
                    summary['ocr_text_extracted'] += 1

                # Charts detected
                chart_analysis = analysis.get('chart_analysis', {})
                if chart_analysis.get('chart_type', 'unknown') != 'unknown':
                    summary['charts_detected'] += 1

                # AI actionable insights
                ai_prompts = knowledge_ext.get('ai_prompt_suggestions', [])
                summary['ai_actionable_insights'].extend(ai_prompts)

        # Top concepts
        from collections import Counter
        concept_freq = Counter(all_concepts)
        summary['key_concepts_extracted'] = [
            {'concept': concept, 'frequency': freq}
            for concept, freq in concept_freq.most_common(20)
        ]

        return summary

    except Exception as e:
        logging.warning(f"Error generating visual insights summary: {e}")
        return summary

def generate_enhanced_ai_insights(total_insights):
    """Generate comprehensive AI insights from all analysis results."""
    insights = {
        'generation_timestamp': str(datetime.datetime.now()),
        'multi_modal_insights': {},
        'cross_content_connections': [],
        'learning_optimization_recommendations': [],
        'content_enhancement_priorities': [],
        'ai_prompt_suggestions': [],
        'knowledge_completeness_score': 0.0
    }

    try:
        # Multi-modal insights
        visual_data = total_insights.get('visual_analysis', {})
        knowledge_graph_data = total_insights.get('knowledge_graph', {})

        if visual_data and isinstance(visual_data, dict) and 'error' not in visual_data:
            insights['multi_modal_insights']['visual'] = {
                'images_analyzed': len(visual_data),
                'concepts_extracted': len(set([
                    concept['concept']
                    for analysis in visual_data.values()
                    if isinstance(analysis, dict)
                    for concept in analysis.get('knowledge_extraction', {}).get('key_concepts', [])
                ])),
                'educational_value': 'high' if len(visual_data) > 10 else 'medium'
            }

        if knowledge_graph_data and isinstance(knowledge_graph_data, dict) and 'error' not in knowledge_graph_data:
            insights['multi_modal_insights']['knowledge_graph'] = {
                'total_nodes': knowledge_graph_data.get('total_nodes', 0),
                'total_edges': knowledge_graph_data.get('total_edges', 0),
                'key_concepts_identified': len(knowledge_graph_data.get('ai_insights', {}).get('key_concepts', [])),
                'learning_paths_available': len(knowledge_graph_data.get('ai_insights', {}).get('learning_paths', []))
            }

        # Cross-content connections
        if knowledge_graph_data.get('cross_connections'):
            insights['cross_content_connections'] = knowledge_graph_data['cross_connections'][:10]  # Top 10

        # Learning optimization recommendations
        recommendations = []

        if visual_data:
            chart_count = sum(1 for analysis in visual_data.values()
                            if isinstance(analysis, dict) and
                            analysis.get('chart_analysis', {}).get('chart_type', 'unknown') != 'unknown')
            if chart_count > 0:
                recommendations.append({
                    'type': 'visual_learning',
                    'priority': 'high',
                    'recommendation': f'Leverage {chart_count} identified charts/diagrams for visual learning enhancement',
                    'action': 'Create interactive explanations for visual content'
                })

        if knowledge_graph_data.get('ai_insights', {}).get('knowledge_gaps'):
            gaps = knowledge_graph_data['ai_insights']['knowledge_gaps']
            if gaps:
                recommendations.append({
                    'type': 'knowledge_completion',
                    'priority': 'medium',
                    'recommendation': f'Address {len(gaps)} identified knowledge gaps',
                    'action': 'Create content to connect isolated concepts'
                })

        insights['learning_optimization_recommendations'] = recommendations

        # AI prompt suggestions
        all_prompts = []

        # From visual analysis
        for analysis in visual_data.values() if visual_data else []:
            if isinstance(analysis, dict):
                prompts = analysis.get('knowledge_extraction', {}).get('ai_prompt_suggestions', [])
                all_prompts.extend(prompts)

        # From knowledge graph
        if knowledge_graph_data.get('ai_insights', {}).get('ai_prompts'):
            all_prompts.extend(knowledge_graph_data['ai_insights']['ai_prompts'])

        insights['ai_prompt_suggestions'] = list(set(all_prompts))[:15]  # Top 15 unique prompts

        # Knowledge completeness score
        completeness_factors = []

        if visual_data:
            completeness_factors.append(min(len(visual_data) / 20, 1.0))  # Visual content coverage

        if knowledge_graph_data.get('total_nodes', 0) > 0:
            completeness_factors.append(min(knowledge_graph_data['total_nodes'] / 100, 1.0))  # Concept coverage

        if completeness_factors:
            insights['knowledge_completeness_score'] = sum(completeness_factors) / len(completeness_factors)

        return insights

    except Exception as e:
        logging.warning(f"Error generating enhanced AI insights: {e}")
        return insights

def generate_ai_prompt_bundle(enhanced_insights):
    """Generate a comprehensive AI prompt bundle for content enhancement."""
    prompt_sections = []

    # Header
    prompt_sections.append("# Enhanced AI Prompt Bundle for Content Optimization\n")
    prompt_sections.append(f"Generated: {enhanced_insights.get('generation_timestamp', 'Unknown')}\n")

    # Visual Content Enhancement
    visual_insights = enhanced_insights.get('multi_modal_insights', {}).get('visual', {})
    if visual_insights:
        prompt_sections.append("## 🔍 Visual Content Enhancement Prompts\n")

        if visual_insights.get('concepts_extracted', 0) > 0:
            prompt_sections.append(f"**Visual Concept Integration**: You have {visual_insights['concepts_extracted']} concepts extracted from visual content. Create comprehensive learning modules that integrate these visual concepts with textual explanations.\n")

        prompt_sections.append("**Chart and Diagram Analysis**: Analyze the extracted charts and diagrams to create step-by-step explanations that guide learners through the visual information systematically.\n")

        prompt_sections.append("**Visual Learning Pathways**: Design learning sequences that progress from simple visual concepts to complex integrated understanding using the identified visual elements.\n")

    # Knowledge Graph Insights
    kg_insights = enhanced_insights.get('multi_modal_insights', {}).get('knowledge_graph', {})
    if kg_insights:
        prompt_sections.append("\n## 🧠 Knowledge Graph Enhancement Prompts\n")

        if kg_insights.get('key_concepts_identified', 0) > 0:
            prompt_sections.append(f"**Concept Relationship Mapping**: Use the {kg_insights['key_concepts_identified']} identified key concepts to create detailed explanations of how these concepts interconnect and build upon each other.\n")

        if kg_insights.get('learning_paths_available', 0) > 0:
            prompt_sections.append(f"**Adaptive Learning Sequences**: Develop {kg_insights['learning_paths_available']} different learning pathways based on the knowledge graph structure, allowing for personalized learning experiences.\n")

    # Cross-Content Connections
    cross_connections = enhanced_insights.get('cross_content_connections', [])
    if cross_connections:
        prompt_sections.append("\n## 🔗 Cross-Content Integration Prompts\n")
        prompt_sections.append(f"**Content Synthesis**: Create unified learning experiences that connect {len(cross_connections)} related content pieces identified through semantic analysis.\n")
        prompt_sections.append("**Concept Reinforcement**: Develop activities that reinforce learning by drawing connections between similar concepts across different content sources.\n")

    # Learning Optimization
    recommendations = enhanced_insights.get('learning_optimization_recommendations', [])
    if recommendations:
        prompt_sections.append("\n## 🎯 Learning Optimization Prompts\n")
        for i, rec in enumerate(recommendations[:5], 1):
            prompt_sections.append(f"**Optimization {i} - {rec.get('type', 'General').title()}**: {rec.get('recommendation', 'Optimize learning experience')} - {rec.get('action', 'Take appropriate action')}\n")

    # Specific AI Prompts
    ai_prompts = enhanced_insights.get('ai_prompt_suggestions', [])
    if ai_prompts:
        prompt_sections.append("\n## 🤖 Specific AI Enhancement Prompts\n")
        for i, prompt in enumerate(ai_prompts[:10], 1):
            prompt_sections.append(f"{i}. {prompt}\n")

    # Knowledge Completeness
    completeness_score = enhanced_insights.get('knowledge_completeness_score', 0)
    prompt_sections.append(f"\n## 📊 Content Completeness Assessment\n")
    prompt_sections.append(f"**Current Completeness Score**: {completeness_score:.2f}/1.0\n")

    if completeness_score < 0.8:
        prompt_sections.append("**Completeness Enhancement**: Focus on filling identified knowledge gaps and creating more comprehensive coverage of key concepts to improve overall learning effectiveness.\n")
    else:
        prompt_sections.append("**Quality Refinement**: With high content completeness, focus on refining the quality and depth of existing materials for optimal learning outcomes.\n")

    return "\n".join(prompt_sections)

def generate_final_summary(total_insights, output_dir):
    """Generate a final summary of all enhanced knowledge extraction results."""
    summary = {
        'extraction_timestamp': str(datetime.datetime.now()),
        'analysis_summary': {
            'visual_content_analyzed': len(total_insights.get('visual_analysis', {})),
            'knowledge_graph_nodes': total_insights.get('knowledge_graph', {}).get('total_nodes', 0),
            'knowledge_graph_edges': total_insights.get('knowledge_graph', {}).get('total_edges', 0),
            'extraction_success': True
        },
        'business_value': {
            'content_optimization_opportunities': 0,
            'learning_path_enhancements': 0,
            'visual_learning_assets': 0,
            'concept_relationship_insights': 0
        },
        'next_steps': []
    }

    try:
        # Calculate business value metrics
        visual_data = total_insights.get('visual_analysis', {})
        if visual_data and isinstance(visual_data, dict):
            summary['business_value']['visual_learning_assets'] = len([
                analysis for analysis in visual_data.values()
                if isinstance(analysis, dict) and
                analysis.get('educational_classification', {}).get('content_category', 'unknown') != 'unknown'
            ])

        kg_data = total_insights.get('knowledge_graph', {})
        if kg_data and isinstance(kg_data, dict):
            ai_insights = kg_data.get('ai_insights', {})
            summary['business_value']['learning_path_enhancements'] = len(ai_insights.get('learning_paths', []))
            summary['business_value']['concept_relationship_insights'] = len(ai_insights.get('key_concepts', []))

        # Next steps
        summary['next_steps'] = [
            "Review generated AI prompt bundle for content enhancement opportunities",
            "Implement visual learning activities based on extracted image analysis",
            "Utilize knowledge graph insights for adaptive learning path creation",
            "Address identified knowledge gaps through targeted content development",
            "Leverage cross-content connections for integrated learning experiences"
        ]

        # Save summary
        summary_path = Path(output_dir) / "final_extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📋 Final summary saved to: {summary_path}")
        logging.info(f"Final summary saved to: {summary_path}")

    except Exception as e:
        logging.warning(f"Error generating final summary: {e}")

if __name__ == "__main__":
    main()
