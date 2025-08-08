flowchart LR
 subgraph core["OMEN Core"]
    direction TB
        config_core["Config (settings)"]
        logging_core["Logging (get_logger)"]
        models_core["Core Models (MetadataItem, MetadataSource, etc)"]
        state_core["State (StateManager)"]
        batch_core["Batch Processing (batch_process)"]
  end
 subgraph extractors["OMEN Extractors"]
    direction TB
        keboola_ext["KeboolaExtractor"]
  end
 subgraph vectorstore["OMEN Vectorstore"]
    direction TB
        embedding_mod["Embedding Providers\n- OpenAIProvider\n- SentenceTransformerProvider\n(+ get_embedding_provider)"]
        indexer_mod["Indexer\n- QdrantIndexer"]
        search_mod["Vector Search - VectorSearch"]
        vs_models["Vector Model -<br>MetadataDocument, SearchQuery, SearchResult"]
  end
 subgraph ontology["OMEN Ontology"]
    direction TB
        ont_manager["OntologyManager"]
        rdf_store["RDFStore (RDF graph storage)"]
        ont_models["Ontology Models<br>- Entity, EntityType<br>- Relationship, RelationshipType"]
  end
 subgraph api["OMEN API - FastAPI"]
    direction TB
        api_main["FastAPI app"]
        api_search["Search Endpoints <br>(query &amp; document CRUD)"]
        api_ontology["Ontology Endpoints <br>(entity/relationship CRUD,<br>SPARQL query)"]
  end
 subgraph cli["OMEN CLI"]
    direction TB
        cli_extract["omen extract"]
        cli_search["omen search"]
        cli_ontology["omen ontology"]
        cli_api["omen api"]
        cli_config["omen config"]
  end
    cli_extract --> keboola_ext & indexer_mod
    keboola_ext --> state_core & config_core & logging_core
    indexer_mod --> embedding_mod & config_core & state_core & batch_core & logging_core
    cli_search --> search_mod
    search_mod --> indexer_mod
    cli_ontology --> ont_manager
    cli_api --> api_main
    cli_config --> config_core
    api_search --> search_mod
    api_ontology --> ont_manager
    ont_manager --> rdf_store & ont_models
    embedding_mod --> config_core & logging_core
    keboola_ext -- fetch metadata --> Keboola_API[["Keboola Storage API"]]
    embedding_mod -- embed text --> OpenAI_API[["OpenAI Embedding API"]]
    indexer_mod -- index & search vectors --> QdrantDB[("Qdrant Vector DB")]

     Keboola_API:::external
     OpenAI_API:::external
     QdrantDB:::external
    classDef external fill:#f0f0f0,stroke:#333,stroke-dasharray:5 5


