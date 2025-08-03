"""
Query matching module for AI-Driven Document Analysis System
Handles query processing, similarity matching, and answer generation
"""

import numpy as np
import ollama
from pathlib import Path
import traceback
import time
import json
from sklearn.metrics.pairwise import cosine_similarity
import re

from config import (
    TARGET_SENTENCES,
    ENABLE_DETAILED_LOGGING,
    OUTPUT_FOLDER,
    DESCRIPTION_MODEL,
    EMBEDDING_MODEL,
    ANSWER_MODEL,
    PROMPTS,
    MAX_QUERY_VARIATIONS,
    SLIDING_WINDOW_SIZE,
    SHOW_TEXT_PREVIEWS,
    PREVIEW_LENGTH,
)

class QueryMatcher:
    def __init__(self, outputFolder=OUTPUT_FOLDER):
        self.outputFolder = Path(outputFolder)
        if ENABLE_DETAILED_LOGGING:
            print(f"[QUERY_INIT] Query matcher initialized with output folder: {self.outputFolder}")
    
    def preprocessQuery(self, userQuery):
        """Preprocess user query to fix grammar/spelling and split multi-part questions"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[QUERY_PREPROCESSING] Starting query preprocessing...")
            print(f"[QUERY_PREPROCESSING] Original query: '{userQuery}'")
            print(f"[QUERY_PREPROCESSING] Using model: {DESCRIPTION_MODEL}")
        
        try:
            # Build the prompt to send to the LLM
            prompt = PROMPTS['query_preprocessing']['user_template'].format(
                user_query=userQuery
            )
            
            if ENABLE_DETAILED_LOGGING and SHOW_TEXT_PREVIEWS:
                print(f"[QUERY_PREPROCESSING] Prompt: '{prompt[:300]}...'")
            
            # Set up the conversation messages
            messages = [
                {'role': 'system', 'content': PROMPTS['query_preprocessing']['system']},
                {'role': 'user', 'content': prompt}
            ]
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[QUERY_PREPROCESSING] Sending request to {DESCRIPTION_MODEL}...")
            startTime = time.time()
            response = ollama.chat(model=DESCRIPTION_MODEL, messages=messages)
            endTime = time.time()
            
            responseText = response['message']['content'].strip()
            if ENABLE_DETAILED_LOGGING:
                print(f"[QUERY_PREPROCESSING] ✓ Response received in {endTime - startTime:.2f} seconds")
                if SHOW_TEXT_PREVIEWS:
                    print(f"[QUERY_PREPROCESSING] Raw response: '{responseText}'")
            
            # Parse JSON response - sometimes LLMs add extra text around the JSON
            try:
                # Try to extract JSON from response (in case there's extra text)
                jsonStart = responseText.find('[')
                jsonEnd = responseText.rfind(']') + 1
                if jsonStart != -1 and jsonEnd != 0:
                    jsonText = responseText[jsonStart:jsonEnd]
                else:
                    jsonText = responseText
                
                cleanedQueries = json.loads(jsonText)
                
                # Validate that we got a list of strings
                if not isinstance(cleanedQueries, list):
                    raise ValueError("Response is not a list")
                
                # Clean up any remaining issues with the queries
                cleanedQueries = [str(query).strip() for query in cleanedQueries if query and str(query).strip()]
                
                if not cleanedQueries:
                    raise ValueError("No valid queries extracted")
                
                if ENABLE_DETAILED_LOGGING:
                    print(f"[QUERY_PREPROCESSING] ✓ Successfully parsed {len(cleanedQueries)} queries:")
                    for i, query in enumerate(cleanedQueries):
                        print(f"[QUERY_PREPROCESSING]   {i+1}: '{query}'")
                
                print(f"[QUERY_PREPROCESSING] ✓ Query preprocessing complete: {len(cleanedQueries)} queries generated")
                return cleanedQueries
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[QUERY_PREPROCESSING] ✗ Error parsing JSON response: {e}")
                if ENABLE_DETAILED_LOGGING:
                    print(f"[QUERY_PREPROCESSING] Problematic response: '{responseText}'")
                print(f"[QUERY_PREPROCESSING] Falling back to original query")
                return [userQuery]
        
        except Exception as e:
            print(f"[QUERY_PREPROCESSING] ✗ Error during query preprocessing: {e}")
            if ENABLE_DETAILED_LOGGING:
                traceback.print_exc()
            print(f"[QUERY_PREPROCESSING] Falling back to original query")
            return [userQuery]
    
    def getAvailableDocuments(self):
        """Get list of processed documents - looks for folders with embeddings files"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[DOC_DISCOVERY] Scanning for processed documents...")
            print(f"[DOC_DISCOVERY] Looking in: {self.outputFolder}")
        
        if not self.outputFolder.exists():
            if ENABLE_DETAILED_LOGGING:
                print(f"[DOC_DISCOVERY] ✗ Output folder does not exist")
            return []
        
        documents = []
        for item in self.outputFolder.iterdir():
            if item.is_dir():
                embeddingsFile = item / "embeddings.npy"
                if ENABLE_DETAILED_LOGGING:
                    print(f"[DOC_DISCOVERY] Checking directory: {item.name}")
                    print(f"[DOC_DISCOVERY]   Embeddings file exists: {embeddingsFile.exists()}")
                
                if embeddingsFile.exists():
                    documents.append(item.name)
                    if ENABLE_DETAILED_LOGGING:
                        print(f"[DOC_DISCOVERY]   ✓ Added to available documents")
                else:
                    if ENABLE_DETAILED_LOGGING:
                        print(f"[DOC_DISCOVERY]   ✗ Missing embeddings file")
        
        print(f"[DOC_DISCOVERY] Found {len(documents)} processed documents: {documents}")
        return documents
    
    def loadAllDocumentEmbeddings(self):
        """Load embeddings from all available documents - combines them into one big array"""
        print(f"[ALL_DOCS_LOAD] Loading embeddings from all processed documents...")
        
        availableDocs = self.getAvailableDocuments()
        if not availableDocs:
            print(f"[ALL_DOCS_LOAD] ✗ No processed documents found")
            return None, None
        
        allEmbeddings = []
        docSentenceMap = []  # Maps (docName, sentenceIndex) to global index
        
        # Load embeddings from each document
        for docName in availableDocs:
            docPath = self.outputFolder / docName
            embeddingsPath = docPath / "embeddings.npy"
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[ALL_DOCS_LOAD] Loading embeddings from: {docName}")
            
            try:
                embeddings = np.load(embeddingsPath)
                if ENABLE_DETAILED_LOGGING:
                    print(f"[ALL_DOCS_LOAD]   ✓ Loaded {len(embeddings)} embeddings")
                
                # Add to global embeddings list
                startIdx = len(allEmbeddings)
                allEmbeddings.extend(embeddings)
                
                # Map each sentence to its document and local index
                for localIdx in range(len(embeddings)):
                    docSentenceMap.append((docName, localIdx))
                
            except Exception as e:
                print(f"[ALL_DOCS_LOAD]   ✗ Error loading embeddings from {docName}: {e}")
                continue
        
        if not allEmbeddings:
            print(f"[ALL_DOCS_LOAD] ✗ No embeddings loaded from any document")
            return None, None
        
        allEmbeddings = np.array(allEmbeddings)
        print(f"[ALL_DOCS_LOAD] ✓ Loaded total of {len(allEmbeddings)} embeddings from {len(availableDocs)} documents")
        
        return allEmbeddings, docSentenceMap
    
    def extractPageNumber(self, content):
        """Extract page number from sentence file content - handles different formats"""
        try:
            # Look for "Page Number: X" pattern first
            pageMatch = re.search(r'Page Number:\s*(\d+)', content)
            if pageMatch:
                return int(pageMatch.group(1))
            else:
                # Fallback: look for just numbers after "Page"
                pageMatch = re.search(r'Page\s*(\d+)', content, re.IGNORECASE)
                if pageMatch:
                    return int(pageMatch.group(1))
                return None
        except:
            return None
    
    def loadSpecificSentences(self, sentenceLocations):
        """Load specific sentences from multiple documents based on locations with metadata"""
        print(f"[MULTI_SENTENCE_LOAD] Loading {len(sentenceLocations)} sentences from multiple documents...")
        
        sentences = []
        sentencesWithMetadata = []
        
        # Load each requested sentence
        for docName, sentenceIdx in sentenceLocations:
            docPath = self.outputFolder / docName
            sentenceFile = docPath / f"sentence_{sentenceIdx + 1}.txt"  # +1 because files are 1-indexed
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[MULTI_SENTENCE_LOAD] Loading: {docName}/sentence_{sentenceIdx + 1}.txt")
            
            try:
                with open(sentenceFile, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    if len(lines) >= 3:
                        sentence = lines[1]  # The sentence is on the second line
                        
                        # Extract page number from content
                        pageNumber = self.extractPageNumber(content)
                        
                        # Create metadata dict with all the useful information
                        metadata = {
                            'document': docName,
                            'sentenceIndex': sentenceIdx,
                            'pageNumber': pageNumber
                        }
                        
                        sentences.append(sentence)
                        sentencesWithMetadata.append({
                            'sentence': sentence,
                            'metadata': metadata
                        })
                        
                        if ENABLE_DETAILED_LOGGING:
                            pageInfo = f"Page {pageNumber}" if pageNumber else "Unknown page"
                            print(f"[MULTI_SENTENCE_LOAD]   ✓ Loaded from {docName} ({pageInfo})")
                            if SHOW_TEXT_PREVIEWS:
                                sentencePreview = sentence[:100] + "..." if len(sentence) > 100 else sentence
                                print(f"[MULTI_SENTENCE_LOAD]   Content: '{sentencePreview}'")
                    else:
                        print(f"[MULTI_SENTENCE_LOAD]   ✗ Invalid file format: {sentenceFile.name}")
                        sentences.append("")
                        sentencesWithMetadata.append({
                            'sentence': "",
                            'metadata': {
                                'document': docName,
                                'sentenceIndex': sentenceIdx,
                                'pageNumber': None
                            }
                        })
                        
            except Exception as e:
                print(f"[MULTI_SENTENCE_LOAD]   ✗ Error loading {sentenceFile.name}: {e}")
                sentences.append("")
                sentencesWithMetadata.append({
                    'sentence': "",
                    'metadata': {
                        'document': docName,
                        'sentenceIndex': sentenceIdx,
                        'pageNumber': None
                    }
                })
        
        print(f"[MULTI_SENTENCE_LOAD] ✓ Loaded {len(sentences)} sentences from multiple documents")
        return sentences, sentencesWithMetadata
    
    def generateQueryVariations(self, userQuery):
        """Generate query variations using LLM - helps find more relevant content"""
        if MAX_QUERY_VARIATIONS == 0:
            print(f"[QUERY_VARIATIONS] No query variations requested (MAX_QUERY_VARIATIONS is set to 0)")
            return [userQuery]
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[QUERY_VARIATIONS] Generating query variations...")
            print(f"[QUERY_VARIATIONS] Original query: '{userQuery}'")
            print(f"[QUERY_VARIATIONS] Using model: {DESCRIPTION_MODEL}")
        
        try:
            # Build the prompt for generating variations
            prompt = PROMPTS['query_variations']['user_template'].format(
                max_variations=MAX_QUERY_VARIATIONS,
                user_query=userQuery
            )
            
            if ENABLE_DETAILED_LOGGING and SHOW_TEXT_PREVIEWS:
                print(f"[QUERY_VARIATIONS] Prompt: '{prompt}'")
            
            # Set up the conversation messages
            messages = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': PROMPTS['query_variations']['assistant'].format(max_variations=MAX_QUERY_VARIATIONS)},
                {'role': 'user', 'content': PROMPTS['query_variations']['instruction'].format(max_variations=MAX_QUERY_VARIATIONS)}
            ]
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[QUERY_VARIATIONS] Sending request to {DESCRIPTION_MODEL}...")
            startTime = time.time()
            response = ollama.chat(model=DESCRIPTION_MODEL, messages=messages)
            endTime = time.time()
            
            variationsText = response['message']['content']
            if ENABLE_DETAILED_LOGGING:
                print(f"[QUERY_VARIATIONS] ✓ Response received in {endTime - startTime:.2f} seconds")
                if SHOW_TEXT_PREVIEWS:
                    print(f"[QUERY_VARIATIONS] Raw response: '{variationsText}'")
            
            # Parse variations (split by lines and clean up)
            variations = [line.strip() for line in variationsText.split('\n') if line.strip()]
            if ENABLE_DETAILED_LOGGING:
                print(f"[QUERY_VARIATIONS] Parsed variations: {variations}")
            
            # Include original query plus variations
            allQueries = [userQuery] + variations[:MAX_QUERY_VARIATIONS]
            
            print(f"[QUERY_VARIATIONS] ✓ Final query set ({len(allQueries)} queries):")
            for i, query in enumerate(allQueries):
                print(f"[QUERY_VARIATIONS]   {i+1}: '{query}'")
            
            return allQueries
        
        except Exception as e:
            print(f"[QUERY_VARIATIONS] ✗ Error generating query variations: {e}")
            if ENABLE_DETAILED_LOGGING:
                traceback.print_exc()
            print(f"[QUERY_VARIATIONS] Falling back to original query only")
            return [userQuery]
    
    def applySlidingWindow(self, similarities, windowSize=SLIDING_WINDOW_SIZE):
        """Apply sliding window convolution with weighted averaging - helps find contextually relevant sentences"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[SLIDING_WINDOW] Applying sliding window convolution...")
            print(f"[SLIDING_WINDOW] Input similarities shape: {similarities.shape}")
            print(f"[SLIDING_WINDOW] Window size: {windowSize}")
            print(f"[SLIDING_WINDOW] Similarities range: [{similarities.min():.4f}, {similarities.max():.4f}]")
        
        if len(similarities) < windowSize:
            if ENABLE_DETAILED_LOGGING:
                print(f"[SLIDING_WINDOW] ⚠ Document corpus too short for window size, returning original similarities")
            return similarities
        
        # Create weights: increasing to center, then decreasing - gives more importance to nearby sentences
        center = windowSize // 2
        weights = []
        for i in range(windowSize):
            if i <= center:
                weights.append(i + 1)
            else:
                weights.append(windowSize - i)
        
        weights = np.array(weights) / sum(weights)  # Normalize weights
        if ENABLE_DETAILED_LOGGING:
            print(f"[SLIDING_WINDOW] Window weights: {weights}")
        
        windowedScores = []
        for i in range(len(similarities)):
            # Define window boundaries
            start = max(0, i - center)
            end = min(len(similarities), i + center + 1)
            
            # Get window data
            windowSimilarities = similarities[start:end]
            windowWeights = weights[:len(windowSimilarities)]
            
            # Calculate weighted average
            weightedScore = np.average(windowSimilarities, weights=windowWeights)
            windowedScores.append(weightedScore)
            
            # Debug first few windows to make sure it's working
            if i < 5 and ENABLE_DETAILED_LOGGING:
                print(f"[SLIDING_WINDOW] Window {i+1}: indices {start}-{end-1}, score {weightedScore:.4f}")
        
        windowedScores = np.array(windowedScores)
        if ENABLE_DETAILED_LOGGING:
            print(f"[SLIDING_WINDOW] ✓ Windowed scores range: [{windowedScores.min():.4f}, {windowedScores.max():.4f}]")
            print(f"[SLIDING_WINDOW] Score improvement (max): {windowedScores.max() - similarities.max():.4f}")
        
        return windowedScores
    
    def findRelevantSentencesAcrossDocs(self, queryEmbeddings, allEmbeddings, docSentenceMap, targetSentences=TARGET_SENTENCES):
        """Find relevant sentences across all documents using dynamic threshold - the core search algorithm"""
        print(f"[DYNAMIC_RELEVANCE_SEARCH] Finding relevant sentences across all documents...")
        print(f"[DYNAMIC_RELEVANCE_SEARCH] Query embeddings: {len(queryEmbeddings)}")
        print(f"[DYNAMIC_RELEVANCE_SEARCH] Total document embeddings: {len(allEmbeddings)}")
        print(f"[DYNAMIC_RELEVANCE_SEARCH] Target sentences: {targetSentences}")
        
        allSimilarities = []
        
        # Calculate similarities for each query variation
        for i, queryEmb in enumerate(queryEmbeddings):
            if ENABLE_DETAILED_LOGGING:
                print(f"[DYNAMIC_RELEVANCE_SEARCH] Processing query variation {i+1}/{len(queryEmbeddings)}")
            
            queryEmb = np.array(queryEmb).reshape(1, -1)
            if ENABLE_DETAILED_LOGGING:
                print(f"[DYNAMIC_RELEVANCE_SEARCH]   Query embedding shape: {queryEmb.shape}")
            
            similarities = cosine_similarity(queryEmb, allEmbeddings)[0]
            if ENABLE_DETAILED_LOGGING:
                print(f"[DYNAMIC_RELEVANCE_SEARCH]   Similarities range: [{similarities.min():.4f}, {similarities.max():.4f}]")
                print(f"[DYNAMIC_RELEVANCE_SEARCH]   Similarities mean: {similarities.mean():.4f}")
            
            allSimilarities.append(similarities)
        
        # Average similarities across all query variations
        if ENABLE_DETAILED_LOGGING:
            print(f"[DYNAMIC_RELEVANCE_SEARCH] Averaging similarities across {len(allSimilarities)} query variations...")
        avgSimilarities = np.mean(allSimilarities, axis=0)
        if ENABLE_DETAILED_LOGGING:
            print(f"[DYNAMIC_RELEVANCE_SEARCH] Averaged similarities range: [{avgSimilarities.min():.4f}, {avgSimilarities.max():.4f}]")
        
        # Apply sliding window convolution to consider context
        if ENABLE_DETAILED_LOGGING:
            print(f"[DYNAMIC_RELEVANCE_SEARCH] Applying sliding window convolution...")
        windowedSimilarities = self.applySlidingWindow(avgSimilarities)
        
        # Dynamic threshold approach: start at 100% and decrease by 1% until we get enough sentences
        print(f"[DYNAMIC_RELEVANCE_SEARCH] Starting dynamic threshold search...")
        currentThreshold = 1.0  # Start at 100% similarity
        thresholdDecrement = 0.01  # Decrease by 1% each iteration
        minThreshold = 0.0  # Minimum threshold to prevent infinite loop
        
        relevantIndices = []
        attempts = 0
        maxAttempts = 100  # Safety limit
        
        while len(relevantIndices) < targetSentences and currentThreshold >= minThreshold and attempts < maxAttempts:
            attempts += 1
            
            # Find sentences above current threshold
            relevantIndices = np.where(windowedSimilarities >= currentThreshold)[0]
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[DYNAMIC_RELEVANCE_SEARCH] Attempt {attempts}: Threshold {currentThreshold:.3f} -> {len(relevantIndices)} sentences")
            
            # If we have enough sentences, break
            if len(relevantIndices) >= targetSentences:
                print(f"[DYNAMIC_RELEVANCE_SEARCH] ✓ Found {len(relevantIndices)} sentences at threshold {currentThreshold:.3f}")
                break
            
            # Lower the threshold
            currentThreshold -= thresholdDecrement
        
        # If we still don't have enough sentences, take the top ones
        if len(relevantIndices) < targetSentences:
            print(f"[DYNAMIC_RELEVANCE_SEARCH] Threshold reached minimum, selecting top {targetSentences} sentences...")
            relevantIndices = np.argsort(windowedSimilarities)[-targetSentences:]
            if ENABLE_DETAILED_LOGGING:
                print(f"[DYNAMIC_RELEVANCE_SEARCH] Top {targetSentences} sentence indices: {relevantIndices}")
        
        # Sort by similarity score (descending) - best matches first
        relevantIndices = relevantIndices[np.argsort(windowedSimilarities[relevantIndices])[::-1]]
        relevantScores = windowedSimilarities[relevantIndices]
        
        # Map global indices back to (docName, sentenceIndex) pairs
        relevantLocations = [docSentenceMap[i] for i in relevantIndices]
        
        print(f"[DYNAMIC_RELEVANCE_SEARCH] ✓ Selected {len(relevantIndices)} relevant sentences from multiple documents")
        print(f"[DYNAMIC_RELEVANCE_SEARCH] Final threshold used: {currentThreshold:.3f}")
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[DYNAMIC_RELEVANCE_SEARCH] Selected scores: {[f'{score:.4f}' for score in relevantScores]}")
            print(f"[DYNAMIC_RELEVANCE_SEARCH] Document distribution:")
            docCounts = {}
            for docName, _ in relevantLocations:
                docCounts[docName] = docCounts.get(docName, 0) + 1
            for docName, count in docCounts.items():
                print(f"[DYNAMIC_RELEVANCE_SEARCH]   {docName}: {count} sentences")
        
        return relevantLocations, relevantScores
    
    def generateFinalAnswer(self, originalQuery, processedQueries, allSentencesWithMetadata):
        """Generate final answer using LLM with enhanced metadata for multiple queries"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[FINAL_ANSWER] Generating final answer...")
            print(f"[FINAL_ANSWER] Original query: '{originalQuery}'")
            print(f"[FINAL_ANSWER] Processed queries: {len(processedQueries)}")
            print(f"[FINAL_ANSWER] Total relevant sentences: {len(allSentencesWithMetadata)}")
            print(f"[FINAL_ANSWER] Using model: {ANSWER_MODEL}")
        
        try:
            # Create context with enhanced document attribution including page numbers
            contextLines = []
            for i, sentenceData in enumerate(allSentencesWithMetadata):
                sentence = sentenceData['sentence']
                metadata = sentenceData['metadata']
                
                # Format metadata string with page number
                if metadata['pageNumber'] is not None:
                    sourceInfo = f"Document: {metadata['document']}, Page: {metadata['pageNumber']}"
                else:
                    sourceInfo = f"Document: {metadata['document']}, Page: Unknown"
                
                contextLines.append(f"[{sourceInfo}]: {sentence}")
            
            context = "\n".join(contextLines)
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[FINAL_ANSWER] Context length: {len(context)} characters")
                if SHOW_TEXT_PREVIEWS:
                    print(f"[FINAL_ANSWER] Context preview:")
                    print(f"[FINAL_ANSWER] '{context[:300]}...'")
            
            # Choose appropriate prompt based on whether we have multiple queries
            if len(processedQueries) == 1:
                finalPrompt = PROMPTS['final_answer']['user_template'].format(
                    context=context,
                    user_query=processedQueries[0]
                )
                messages = [
                    {'role': 'user', 'content': processedQueries[0]},
                    {'role': 'assistant', 'content': PROMPTS['final_answer']['context_intro']},
                    {'role': 'user', 'content': finalPrompt}
                ]
            else:
                processedQueriesText = "\n".join([f"{i+1}. {query}" for i, query in enumerate(processedQueries)])
                finalPrompt = PROMPTS['final_answer_multiple']['user_template'].format(
                    context=context,
                    original_query=originalQuery,
                    processed_queries=processedQueriesText
                )
                messages = [
                    {'role': 'user', 'content': originalQuery},
                    {'role': 'assistant', 'content': PROMPTS['final_answer_multiple']['context_intro']},
                    {'role': 'user', 'content': finalPrompt}
                ]
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[FINAL_ANSWER] Final prompt length: {len(finalPrompt)} characters")
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[FINAL_ANSWER] Sending request to {ANSWER_MODEL}...")
            startTime = time.time()
            response = ollama.chat(model=ANSWER_MODEL, messages=messages)
            endTime = time.time()
            
            answer = response['message']['content']
            if ENABLE_DETAILED_LOGGING:
                print(f"[FINAL_ANSWER] ✓ Response received in {endTime - startTime:.2f} seconds")
                print(f"[FINAL_ANSWER] Answer length: {len(answer)} characters")
                if SHOW_TEXT_PREVIEWS:
                    print(f"[FINAL_ANSWER] Answer preview:")
                    print(f"[FINAL_ANSWER] '{answer[:PREVIEW_LENGTH]}...'")
            
            return answer
        
        except Exception as e:
            print(f"[FINAL_ANSWER] ✗ Error generating final answer: {e}")
            if ENABLE_DETAILED_LOGGING:
                traceback.print_exc()
            return "Error generating answer."
    
    def processSingleQuery(self, query, allEmbeddings, docSentenceMap, targetSentences=TARGET_SENTENCES):
        """Process a single query and return relevant sentences with metadata"""
        print(f"[SINGLE_QUERY_PROCESS] Processing query: '{query}'")
        
        # Generate query variations to improve search robustness
        queryVariations = self.generateQueryVariations(query)
        
        # Generate embeddings for all query variations
        queryEmbeddings = []
        for i, variation in enumerate(queryVariations):
            if ENABLE_DETAILED_LOGGING:
                print(f"[SINGLE_QUERY_PROCESS] Generating embedding for variation {i+1}/{len(queryVariations)}: '{variation}'")
            try:
                response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=variation)
                embedding = response['embedding']
                queryEmbeddings.append(embedding)
                if ENABLE_DETAILED_LOGGING:
                    print(f"[SINGLE_QUERY_PROCESS] ✓ Variation {i+1} embedding generated (dimensions: {len(embedding)})")
            except Exception as e:
                print(f"[SINGLE_QUERY_PROCESS] ✗ Error generating embedding for variation '{variation}': {e}")
        
        if not queryEmbeddings:
            print(f"[SINGLE_QUERY_PROCESS] ✗ Failed to generate any query embeddings for '{query}'")
            return [], []
        
        print(f"[SINGLE_QUERY_PROCESS] ✓ Generated {len(queryEmbeddings)} query embeddings")
        
        # Find relevant sentences using our dynamic threshold algorithm
        relevantLocations, scores = self.findRelevantSentencesAcrossDocs(
            queryEmbeddings, allEmbeddings, docSentenceMap, targetSentences
        )
        
        # Load the actual sentences with their metadata
        relevantSentences, sentencesWithMetadata = self.loadSpecificSentences(relevantLocations)
        
        return sentencesWithMetadata, scores
    
    def queryAllDocuments(self, userQuery, targetSentences=TARGET_SENTENCES):
        """Main query processing function across all documents with preprocessing - the main entry point"""
        print(f"\n{'='*80}")
        print(f"[GLOBAL_QUERY_PROCESS] STARTING MULTI-DOCUMENT QUERY WITH PREPROCESSING")
        print(f"[GLOBAL_QUERY_PROCESS] Original query: '{userQuery}'")
        print(f"[GLOBAL_QUERY_PROCESS] Target sentences per query: {targetSentences}")
        print(f"{'='*80}")
        
        # Step 1: Preprocess the query (fix grammar/spelling and split multi-part questions)
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[GLOBAL_QUERY_PROCESS] STEP 1: QUERY PREPROCESSING")
        processedQueries = self.preprocessQuery(userQuery)
        
        # Step 2: Load all document embeddings into memory
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[GLOBAL_QUERY_PROCESS] STEP 2: LOADING ALL DOCUMENT EMBEDDINGS")
        allEmbeddings, docSentenceMap = self.loadAllDocumentEmbeddings()
        if allEmbeddings is None or docSentenceMap is None:
            print(f"[GLOBAL_QUERY_PROCESS] ✗ Failed to load document embeddings")
            return
        
        # Step 3: Process each cleaned query separately
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[GLOBAL_QUERY_PROCESS] STEP 3: PROCESSING INDIVIDUAL QUERIES")
        
        allSentencesWithMetadata = []
        allScores = []
        queryResults = []
        
        for i, query in enumerate(processedQueries):
            print(f"\n[GLOBAL_QUERY_PROCESS] Processing query {i+1}/{len(processedQueries)}: '{query}'")
            
            sentencesWithMetadata, scores = self.processSingleQuery(
                query, allEmbeddings, docSentenceMap, targetSentences
            )
            
            queryResults.append({
                'query': query,
                'sentences': sentencesWithMetadata,
                'scores': scores
            })
            
            # Add to combined results for final answer generation
            allSentencesWithMetadata.extend(sentencesWithMetadata)
            allScores.extend(scores)
            
            print(f"[GLOBAL_QUERY_PROCESS] ✓ Query {i+1} processed: {len(sentencesWithMetadata)} relevant sentences found")
        
        # Step 4: Generate final answer using all results
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[GLOBAL_QUERY_PROCESS] STEP 4: FINAL ANSWER GENERATION")
        answer = self.generateFinalAnswer(userQuery, processedQueries, allSentencesWithMetadata)
        
        # Display results in a nice format
        print(f"\n{'='*80}")
        print(f"[GLOBAL_QUERY_PROCESS] ✓ MULTI-DOCUMENT QUERY PROCESSING COMPLETE!")
        print(f"[GLOBAL_QUERY_PROCESS] Original query: '{userQuery}'")
        print(f"[GLOBAL_QUERY_PROCESS] Processed into {len(processedQueries)} queries")
        print(f"[GLOBAL_QUERY_PROCESS] Total relevant sentences found: {len(allSentencesWithMetadata)}")
        print(f"{'='*80}")
        print("PROCESSED QUERIES:")
        print("="*80)
        for i, query in enumerate(processedQueries):
            print(f"{i+1}. {query}")
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)
        
        print("\n" + "="*80)
        print("RELEVANT CONTEXT (FROM MULTIPLE DOCUMENTS):")
        print("="*80)
        
        # Display results organized by query for easier understanding
        for i, result in enumerate(queryResults):
            query = result['query']
            sentences = result['sentences']
            scores = result['scores']
            
            print(f"\n--- RESULTS FOR QUERY {i+1}: '{query}' ---")
            
            for j, (sentenceData, score) in enumerate(zip(sentences, scores)):
                sentence = sentenceData['sentence']
                metadata = sentenceData['metadata']
                
                # Format display with page number for easy reference
                if metadata['pageNumber'] is not None:
                    sourceInfo = f"Document: {metadata['document']}, Page: {metadata['pageNumber']}, Sentence: {metadata['sentenceIndex']+1}"
                else:
                    sourceInfo = f"Document: {metadata['document']}, Page: Unknown, Sentence: {metadata['sentenceIndex']+1}"
                
                print(f"\n[Q{i+1}-{j+1}] {sourceInfo} (Score: {score:.4f})")
                print(f"{sentence}")
        
        print("="*80)
        
        return answer