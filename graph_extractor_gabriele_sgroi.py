#
#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/
#


# Define the script version in terms of Semantic Versioning (SemVer)
# when Git or other versioning systems are not employed.
__version__ = "0.0.0"
from pathlib import Path
print("'" + Path(__file__).stem + ".py'  v" + __version__)


# Add path 'savvy' so it can be found for library imports
from pathlib import Path
import sys
sys.path.insert(1, str(Path(Path.cwd().parent).joinpath("savvy")))      


from savvy_secrets import api_secrets
if not "api_openai" in api_secrets.keys(): raise Exception("ERROR: api_secrets from savvy_secrets.py doesn't have the key requested.")
# 'api_openai': ['na','OpenAI_api_key', 'organization', 'project'],
OPENAI_API_KEY = api_secrets['api_openai'][1]
OPENAI_ORG = api_secrets['api_openai'][2]
OPENAI_PROJ = api_secrets['api_openai'][3]
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


from savvy_secrets import api_secrets
if not "api_openrouter" in api_secrets.keys(): raise Exception("ERROR: api_secrets from savvy_secrets.py doesn't have the key requested.")
OPENROUTER_API_KEY = api_secrets['api_openrouter'][1]
import os
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY


from savvy_secrets import api_secrets
if not "google_gemini" in api_secrets.keys(): raise Exception("ERROR: google_gemini from savvy_secrets.py doesn't have the key requested.")
GOOGLE_API_KEY = api_secrets['google_gemini'][1]

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"

# Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`

"""
How To Build a Multi-Source Knowledge Graph Extractor from Scratch

https://medium.com/data-science-collective/how-to-build-a-multi-source-knowledge-graph-extractor-from-scratch-60f0a51e17b5

https://github.com/GabrieleSgroi/knowledge_graph_extraction/tree/public

https://colab.research.google.com/drive/1st_E7SBEz5GpwCnzGSvKaVUiQuKv3QGZ#scrollTo=LktBTM8hMQF3


pip install git+https://github.com/GabrieleSgroi/knowledge_graph_extraction.git
pip install matplotlib
"""



def ex_graph_extractor_gabriele_sgroi_build_extract(file=None):
    """
    Create a simple and straightforward agentic workflow implementation to extract 
    and expand consistent Knowledge Graphs from Wikipedia pages.

    Uses a Large Language Model to build and refine the knowledge graph.

    """

    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from wikipedia import wikipedia

    from kg_builder.cfg import SplitConfig
    from kg_builder.engine import GeminiEngine
    from kg_builder.prompts.prompting import ExtractorPrompting, BuilderPrompting

    from kg_builder.workflow import EBWorkflow
    from kg_builder.relations import RelationsData
    import os
    from tqdm import tqdm

    if file is None: raise Exception("Argument 'file' not passed")

    if file.is_file():
        print(f"The knowledge graph already exists as file {file}")
        return file

    if not file.is_file():

        # Rebuild the knowledge graph

        # SET GEMINI API KEY AS ENVIRONMENT VARIABLE
        os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        extractor_engine = GeminiEngine(model_id="gemini-2.0-flash", prompting=ExtractorPrompting())
        builder_engine = GeminiEngine(model_id="gemini-2.0-flash", prompting=BuilderPrompting())

        extractor = EBWorkflow(embeddings=embeddings,
                                split_cfg=SplitConfig(chunk_char_size=4096,
                                                    chunk_char_overlap=256))

        wiki_pages = ['Facebook', 'Instagram', 'WhatsApp',]

        summaries = [wikipedia.page(p, auto_suggest=False).summary for p in wiki_pages]

        # DEFINE DOCUMENTS TO STRUCTURE
        documents = summaries

        # DEFINE THE TYPES OF THE ENTITIES TO EXTRACT
        allowed_entity_types = ["person", "company"]

        relations_data = RelationsData.empty(allowed_entity_types=allowed_entity_types)

        """
        for doc in documents:
            print(doc)
        Facebook is a social media and social networking service owned by the American technology conglomerate Meta. Created in 2004 by Mark Zuckerberg with four other Harvard College students and roommates, Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes, its name derives from the face book directories often given to American university students. Membership was initially limited to Harvard students, gradually expanding to other North American universities. Since 2006, Facebook allows everyone to register from 13 years old, except in the case of a handful of nations, where the age requirement is 14 years. As of December 2023, Facebook claimed almost 3.07 billion monthly active users worldwide. As of November 2024, Facebook ranked as the third-most-visited website in the world, with 23% of its traffic coming from the United States. It was the most downloaded mobile app of the 2010s.
        Facebook can be accessed from devices with Internet connectivity, such as personal computers, tablets and smartphones. After registering, users can create a profile revealing personal information about themselves. They can post text, photos and multimedia which are shared with any other users who have agreed to be their friend or, with different privacy settings, publicly. Users can also communicate directly with each other with Messenger, edit messages (within 15 minutes after sending), join common-interest groups, and receive notifications on the activities of their Facebook friends and the pages they follow.
        Facebook has often been criticized over issues such as user privacy (as with the Facebook-Cambridge Analytica data scandal), political manipulation (as with the 2016 U.S. elections) and mass surveillance. The company has also been subject to criticism over its psychological effects such as addiction and low self-esteem, and over content such as fake news, conspiracy theories, copyright infringement, and hate speech. Commentators have accused Facebook of willingly facilitating the spread of such content, as well as exaggerating its number of users to appeal to advertisers.
        Instagram is an American photo and short-form video sharing social networking service owned by Meta Platforms. It allows users to upload media that can be edited with filters, be organized by hashtags, and be associated with a location via geographical tagging. Posts can be shared publicly or with preapproved followers. Users can browse other users' content by tags and locations, view trending content, like photos, and follow other users to add their content to a personal feed. A Meta-operated image-centric social media platform, it is available on iOS, Android, Windows 10, and the web. Users can take photos and edit them using built-in filters and other tools, then share them on other social media platforms like Facebook. It supports 32 languages including English, Hindi,  Spanish, French, Korean, and Japanese.
        Instagram was originally distinguished by allowing content to be framed only in a square (1:1) aspect ratio of 640 pixels to match the display width of the iPhone at the time. In 2015, this restriction was eased with an increase to 1080 pixels. It also added messaging features, the ability to include multiple images or videos in a single post, and a Stories feature—similar to its main competitor, Snapchat, which allowed users to post their content to a sequential feed, with each post accessible to others for 24 hours. As of January 2019, Stories was used by 500 million people daily.

        Instagram was launched for iOS in October 2010 by Kevin Systrom and Mike Krieger. It rapidly gained popularity, reaching 1 million registered users in two months, 10 million in a year, and 1 billion in June 2018. In April 2012, Facebook acquired the service for approximately US$1 billion in cash and stock. The Android version of Instagram was released in April 2012, followed by a feature-limited desktop interface in November 2012, a Fire OS app in June 2014, and an app for Windows 10 in October 2016. Although often admired for its success and influence, Instagram has also been criticized for negatively affecting teens' mental health, its policy and interface changes, its alleged egalegal and inappropriate content uploaded by users.
        WhatsApp (officially WhatsApp Messenger) is an American social media, instant messaging (IM), and voice-over-IP (VoIP) service owned by technology conglomerate Meta. It allows users to send text, voice messages and video messages, make voice and video calls, and share images, documents, user locations, and other content. WhatsApp's client application runs on mobile devices, and can be accessed from computers. The service requires a cellular mobile telephone number to sign up. WhatsApp was launched in February 2009. In January 2018, WhatsApp released a standalone business app called WhatsApp Business which can communicate with the standard WhatsApp client.
        The service was created by WhatsApp Inc. of Mountain View, California, which was acquired by Facebook in February 2014 for approximately US$19.3 billion. It became the world's most popular messaging application by 2015, and had more than 2 billion users worldwide by February 2020, with WhatsApp Business having approximately 200 million monthly users by 2023. By 2016, it had become the primary means of Internet communication in regions including the Americas, the Indian subcontinent, and large parts of Europe and Africa.
        """

        for doc in tqdm(documents):
            triplets = extractor(text=doc,
                                extractor_engine=extractor_engine,
                                builder_engine=builder_engine,
                                relations_data=relations_data,
                                delete_models_after_use=False)

        # Save a graphical image of the relations and the relations as a JSON file.
        #folder = f'./{datetime.now().strftime("%y%m%d_%H%M%S")}'
        relations_data.save_graph_plot(str(folder))
        print(f"Saved relations graphical plot to {folder}")
        relations_data.save_json(str(file))
        print(f"Saved relations to {file}")

        relations_data.get_undirected_shortest_path(source="mike krieger", target="mark zuckerberg")

        return file



def ex_graph_extractor_gabriele_sgroi_query(file=None, query=None):
    """
    Using the Knowledge Graph file specified as "file", query the knowledge graph
    with the query "query", using LangChain for NER.


    Corrected JSON Parsing: Fixed the RelationsData.load_json() method in relations.py to correctly parse the provided JSON file format, ensuring that the annotated_passages and allowed_entity_types are loaded into the RelationsData object.

    """

    from kg_builder.relations import RelationsData
    import networkx as nx
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.schema import AIMessage

    if file is None or query is None:
        raise Exception("Argument missing!")

    # DEFINE THE TYPES OF THE ENTITIES TO EXTRACT
    allowed_entity_types = ["person", "company"]

    relations_data = RelationsData.load_json(str(file))
    print(f"\nRetrieved relations from {file}")
    #print(relations_data.annotated_passages)

    graph = relations_data.networkx_graph  # Access the NetworkX graph
    print(f"\nNumber of nodes in graph: {len(graph.nodes())}") 
    if len(graph.nodes()) > 0:
        print(f"Sample nodes: {list(graph.nodes())[:5]}")  # Debugging
    else:
        print("Graph has NO nodes!")
        return None

    # 1.  **Entity Recognition using LangChain:**

    # 1.1. Define Prompt for NER (can be refined)
    ner_prompt_template = """
    Extract the entities from the following text.
    Only extract the names of entities belonging to the following types: {entity_types}.
    Return the entities as a comma-separated list.

    Text: {text}
    Entities:
    """
    ner_prompt = PromptTemplate(
        template=ner_prompt_template,
        input_variables=["text", "entity_types"]
    )

    # 1.2.  Create LLM Chain for NER
    from langchain_openai import ChatOpenAI
    try:
        llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            temperature=0,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    except Exception as e:
        raise Exception(e)

    ner_chain = ner_prompt | llm

    def extract_entities(text, entity_types):
        ner_output = ner_chain.invoke({"text": text, "entity_types": entity_types})

        # Check if ner_output is an AIMessage and extract the content
        if isinstance(ner_output, AIMessage):
            ner_output_text = ner_output.content
        else:
            ner_output_text = str(ner_output)  # Ensure it's a string

        return [entity.strip() for entity in ner_output_text.split(",")]

    query_entities = extract_entities(query, ", ".join(allowed_entity_types))
    print(f"Extracted Entities from Query (using LangChain RunnableSequence): {query_entities}")
    if len(query_entities) == 0:
        print(f"No entities were extracted for the query {query}")
        return None

    # 2.  **KG Retrieval:**
    #     Now, use the extracted entities to find relevant information in the KG.
    relevant_triplets = []

    for entity in query_entities:
        entity_lower = entity.strip().lower()  # Normalize the entity
        for node in graph.nodes():
            # This block is never executed because graph.nodes() is empty.
            #print(f"entity: {entity_lower} ~ {node}")
            if entity_lower == node.strip().lower():  # Normalize the node
                # Find all connected triplets (nodes and relations)
                for neighbor in graph.neighbors(node):
                    relation = graph.get_edge_data(node, neighbor).get("relation")
                    relevant_triplets.append((node, relation, neighbor))
                for predecessor in graph.predecessors(node):  # Check incoming edges too
                    relation = graph.get_edge_data(predecessor, node).get("relation")
                    relevant_triplets.append((predecessor, relation, node))
                break  # Stop searching nodes once a match is found

    print(f"{len(relevant_triplets)} relevant connected triplets (nodes and relations) from KG found")
    for triplet in relevant_triplets:
        print(f"\t{triplet}")

    # 3.  **Format for LLM:**
    #     The key is to present the KG information to the LLM in a way it
    #     can understand.  A clear, structured format is best.
    context = "\n".join([f"{s} {p} {o}." for s, p, o in relevant_triplets])
    print(f"\ncontext (data to be passed to an LLM):\n{context}")

    llm_prompt = f"""
    You are a Knowledge Graph answering assistant.
    Use the following information from a Knowledge Graph to answer the user's query.

    Knowledge Graph Context:
    {context}

    User Query:
    {query}

    Answer:
    """
    #print("LLM Prompt:\n", llm_prompt)

    # 4.  **LLM Call (Placeholder):**
    #     This is where you would integrate with your LLM (OpenAI, Gemini, etc.).
    #     I'll leave this as a placeholder, as the actual code depends on the
    #     LLM library you use.

    def call_llm(prompt):
        from openai import OpenAI  
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    answer = call_llm(llm_prompt)
    print(f"\nquestion: {query}")
    print("LLM Answer:", answer)

    return answer


if __name__ == '__main__':
    pass

    #ex_graph_extractor_gabriele_sgroi_wiki_social(rebuild=False)

    from pathlib import Path
    folder = Path(Path.cwd()).joinpath("kg_builder_kg")
    if not folder.is_dir(): raise Exception(f"Folder not found {folder}")
    print(f"folder: {folder}")
    file = folder.joinpath("relations.json")
    if not file.is_file(): raise Exception(f"File not found {file}")
    
    # Uncomment the two below to rebuild or create the Knowledge Graph for the first time.  
    # Once the knowledge graph is built, you only need to run ex_graph_extractor_gabriele_sgroi_query().
    #if file.is_file(): file.unlink()       # Uncomment to rebuild the knowledge graph
    file = ex_graph_extractor_gabriele_sgroi_build_extract(file=file)

    query = "Who launched instagram?"
    answer = ex_graph_extractor_gabriele_sgroi_query(file=file, query=query)

    # -----------------------------------------------------------------------------------------------------------

    # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-