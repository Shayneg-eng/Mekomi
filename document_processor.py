"""
Enhanced Document processing module for AI-Driven Document Analysis System
Handles text extraction, cleaning, sentence splitting, embedding generation, and page number detection
"""

import os
import re
import numpy as np
import ollama
from pathlib import Path
import PyPDF2
import statistics
import traceback
import time
from difflib import SequenceMatcher

from config import (
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    ENABLE_DETAILED_LOGGING,
    DESCRIPTION_MODEL,
    EMBEDDING_MODEL,
    PROMPTS,
    SAMPLE_SENTENCES_FIRST,
    SAMPLE_SENTENCES_INTERVAL,
    MIN_SENTENCE_LENGTH,
    SHOW_TEXT_PREVIEWS,
    PREVIEW_LENGTH,
    SHOW_EMBEDDING_SAMPLES,
    EMBEDDING_SAMPLE_SIZE,
    PROGRESS_UPDATE_INTERVAL,
)

class DocumentProcessor:
    def __init__(self, inputFolder=INPUT_FOLDER, outputFolder=OUTPUT_FOLDER):
        # Initialize the document processor with folder paths
        self.inputFolder = Path(inputFolder)
        self.outputFolder = Path(outputFolder)
        self.setupDirectories()
        # Store pages and full text for page number detection - this helps us figure out which page each sentence came from
        self.documentPages = None
        self.documentFullText = None
    
    def setupDirectories(self):
        """Create necessary directories if they don't exist yet"""
        if ENABLE_DETAILED_LOGGING:
            print("[SETUP] Creating directory structure...")
        self.inputFolder.mkdir(exist_ok=True)
        self.outputFolder.mkdir(exist_ok=True)
        if ENABLE_DETAILED_LOGGING:
            print(f"[SETUP] ✓ Input folder: {self.inputFolder}")
            print(f"[SETUP] ✓ Output folder: {self.outputFolder}")
    
    def extractTextFromPdf(self, pdfPath):
        """Extract text from PDF file and store page information for page number detection"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[PDF_EXTRACT] Starting PDF text extraction from: {pdfPath.name}")
        try:
            with open(pdfPath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if ENABLE_DETAILED_LOGGING:
                    print(f"[PDF_EXTRACT] PDF has {len(reader.pages)} pages")
                
                text = ""
                pages = []
                fullTextParts = []
                
                # Go through each page and extract the text
                for i, page in enumerate(reader.pages):
                    if ENABLE_DETAILED_LOGGING:
                        print(f"[PDF_EXTRACT] Processing page {i+1}/{len(reader.pages)}", end='\r')
                    
                    rawText = page.extract_text()
                    # Apply same normalization as main text processing - keeps everything consistent
                    normalizedText = self.normalizeText(rawText)
                    
                    # Store page information for page number detection later
                    pages.append({
                        'pageNum': i + 1,
                        'text': normalizedText,
                        'rawText': rawText
                    })
                    
                    fullTextParts.append(normalizedText)
                    text += rawText + "\n"
                    
                    # Show a preview of the first page to make sure extraction is working
                    if i == 0 and SHOW_TEXT_PREVIEWS and ENABLE_DETAILED_LOGGING:
                        print(f"\n[PDF_EXTRACT] First page text sample (first {PREVIEW_LENGTH} chars):")
                        print(f"[PDF_EXTRACT] '{rawText[:PREVIEW_LENGTH]}...'")
                
                # Store for page number detection - we'll use this later to figure out which page each sentence is on
                self.documentPages = pages
                self.documentFullText = self.normalizeText(' '.join(fullTextParts))
                
                if ENABLE_DETAILED_LOGGING:
                    print(f"\n[PDF_EXTRACT] ✓ Extracted {len(text)} characters total")
                    print(f"[PDF_EXTRACT] ✓ Stored {len(pages)} pages for page number detection")
                return text
        except Exception as e:
            print(f"[PDF_EXTRACT] ✗ Error reading PDF {pdfPath}: {e}")
            return None
    
    def extractTextFromTxt(self, txtPath):
        """Extract text from TXT file - much simpler than PDF since it's already text"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[TXT_EXTRACT] Starting TXT file reading from: {txtPath.name}")
        try:
            with open(txtPath, 'r', encoding='utf-8') as file:
                text = file.read()
                # For TXT files, create a single page entry since there's no real page concept
                normalizedText = self.normalizeText(text)
                self.documentPages = [{
                    'pageNum': 1,
                    'text': normalizedText,
                    'rawText': text
                }]
                self.documentFullText = normalizedText
                
                if ENABLE_DETAILED_LOGGING:
                    print(f"[TXT_EXTRACT] ✓ Read {len(text)} characters")
                    print(f"[TXT_EXTRACT] ✓ Created single page entry for page number detection")
                    if SHOW_TEXT_PREVIEWS:
                        print(f"[TXT_EXTRACT] Text sample (first {PREVIEW_LENGTH} chars):")
                        print(f"[TXT_EXTRACT] '{text[:PREVIEW_LENGTH]}...'")
                return text
        except Exception as e:
            print(f"[TXT_EXTRACT] ✗ Error reading TXT {txtPath}: {e}")
            return None
    
    def normalizeText(self, text):
        """Normalize text the same way for both processing and page detection - consistency is key"""
        # Clean up extra whitespace and line breaks to make text more uniform
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def cleanText(self, text):
        """Clean and normalize text to make it ready for processing"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[TEXT_CLEAN] Starting text cleaning...")
            print(f"[TEXT_CLEAN] Original text length: {len(text)} characters")
            
            if SHOW_TEXT_PREVIEWS:
                print(f"[TEXT_CLEAN] Original text sample:")
                print(f"[TEXT_CLEAN] '{text[:150]}...'")
        
        # Use the same normalization method to keep everything consistent
        text = self.normalizeText(text)
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[TEXT_CLEAN] ✓ Cleaned text length: {len(text)} characters")
            if SHOW_TEXT_PREVIEWS:
                print(f"[TEXT_CLEAN] Cleaned text sample:")
                print(f"[TEXT_CLEAN] '{text[:150]}...'")
        
        return text
    
    def findSentencePageNumber(self, sentence):
        """Find which page contains a sentence using multiple strategies - this is tricky because OCR can be imperfect"""
        if not self.documentPages or not self.documentFullText:
            if ENABLE_DETAILED_LOGGING:
                print(f"[PAGE_DETECT] No page information available")
            return None
        
        cleanSentence = self.normalizeText(sentence)
        
        # Remove trailing period for matching - sometimes periods get lost in OCR
        if cleanSentence.endswith('.'):
            cleanSentence = cleanSentence[:-1]
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[PAGE_DETECT] Finding page for sentence: '{cleanSentence[:50]}...'")
        
        # Strategy 1: Direct text search in individual pages - the most reliable method
        for pageInfo in self.documentPages:
            if cleanSentence in pageInfo['text']:
                if ENABLE_DETAILED_LOGGING:
                    print(f"[PAGE_DETECT] ✓ Direct match on page {pageInfo['pageNum']}")
                return pageInfo['pageNum']
        
        # Strategy 2: Fuzzy matching for OCR errors - sometimes text gets slightly mangled during extraction
        bestMatch = None
        bestRatio = 0.0
        threshold = 0.8  # Need pretty high similarity to be confident
        
        for pageInfo in self.documentPages:
            # Skip very short pages or very long sentences on short pages - these matches would be unreliable
            if len(pageInfo['text']) < 50 or len(cleanSentence) > len(pageInfo['text']) * 0.8:
                continue
                
            ratio = SequenceMatcher(None, cleanSentence, pageInfo['text']).ratio()
            if ratio > bestRatio and ratio > threshold:
                bestRatio = ratio
                bestMatch = pageInfo['pageNum']
        
        if bestMatch:
            if ENABLE_DETAILED_LOGGING:
                print(f"[PAGE_DETECT] ✓ Fuzzy match on page {bestMatch} (ratio: {bestRatio:.3f})")
            return bestMatch
        
        # Strategy 3: Look for sentence fragments - sometimes we can find the beginning and end
        words = cleanSentence.split()
        if len(words) > 5:  # Only for longer sentences where this makes sense
            # Try first and last few words
            startFragment = ' '.join(words[:3])
            endFragment = ' '.join(words[-3:])
            
            for pageInfo in self.documentPages:
                if startFragment in pageInfo['text'] and endFragment in pageInfo['text']:
                    if ENABLE_DETAILED_LOGGING:
                        print(f"[PAGE_DETECT] ✓ Fragment match on page {pageInfo['pageNum']}")
                    return pageInfo['pageNum']
        
        # Strategy 4: Position-based estimation using full text - last resort but sometimes works
        try:
            pos = self.documentFullText.find(cleanSentence)
            if pos != -1:
                # Estimate page based on position in the full text
                charsPerPage = len(self.documentFullText) / len(self.documentPages)
                estimatedPage = min(int(pos / charsPerPage) + 1, len(self.documentPages))
                
                if ENABLE_DETAILED_LOGGING:
                    print(f"[PAGE_DETECT] Position-based estimate: page {estimatedPage}")
                
                # Check the estimated page and neighbors - the estimate might be slightly off
                for pageOffset in [0, -1, 1, -2, 2]:
                    checkPage = estimatedPage + pageOffset
                    if 1 <= checkPage <= len(self.documentPages):
                        pageIdx = checkPage - 1
                        if cleanSentence in self.documentPages[pageIdx]['text']:
                            if ENABLE_DETAILED_LOGGING:
                                print(f"[PAGE_DETECT] ✓ Position verification found on page {checkPage}")
                            return checkPage
        except:
            # If anything goes wrong with position estimation, just continue
            pass
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[PAGE_DETECT] ✗ No page found for sentence")
        return None
    
    def splitIntoSentences(self, text):
        """Split text into sentences using regex - not perfect but pretty good for most documents"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[SENTENCE_SPLIT] Starting sentence tokenization...")
            print(f"[SENTENCE_SPLIT] Input text length: {len(text)} characters")
        
        # Split using regex - looks for periods, exclamation marks, or question marks followed by whitespace
        rawSentences = re.split(r'[.!?]+\s+', text)
        if ENABLE_DETAILED_LOGGING:
            print(f"[SENTENCE_SPLIT] Raw split resulted in {len(rawSentences)} segments")
        
        cleanedSentences = []
        for i, sentence in enumerate(rawSentences):
            sentence = sentence.strip()
            # Only keep sentences that are long enough to be meaningful
            if len(sentence) > MIN_SENTENCE_LENGTH:
                # Add period if it doesn't already end with punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                cleanedSentences.append(sentence)
                
                # Show first few sentences for debugging - helps verify the splitting is working
                if i < 3 and ENABLE_DETAILED_LOGGING and SHOW_TEXT_PREVIEWS:
                    print(f"[SENTENCE_SPLIT] Sentence {i+1}: '{sentence[:100]}...'")
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[SENTENCE_SPLIT] ✓ Final sentence count: {len(cleanedSentences)}")
            print(f"[SENTENCE_SPLIT] Average sentence length: {np.mean([len(s) for s in cleanedSentences]):.1f} chars")
        
        return cleanedSentences
    
    def sampleSentencesForLlm(self, sentences):
        """Sample sentences for LLM description - we can't send the whole document to the LLM"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[LLM_SAMPLING] Creating sentence samples for document description...")
            print(f"[LLM_SAMPLING] Total sentences available: {len(sentences)}")
        
        sampled = []
        
        # First N sentences - these usually contain important intro information
        firstCount = min(SAMPLE_SENTENCES_FIRST, len(sentences))
        if ENABLE_DETAILED_LOGGING:
            print(f"[LLM_SAMPLING] Taking first {firstCount} sentences:")
        for i in range(firstCount):
            if ENABLE_DETAILED_LOGGING and SHOW_TEXT_PREVIEWS:
                sentencePreview = sentences[i][:100] + "..." if len(sentences[i]) > 100 else sentences[i]
                print(f"[LLM_SAMPLING]   {i+1}: '{sentencePreview}'")
            sampled.append(f"Sentence {i+1}: \"{sentences[i]}\"")
        
        # Every Nth sentence - gives us a sampling throughout the document
        if len(sentences) > SAMPLE_SENTENCES_INTERVAL:
            if ENABLE_DETAILED_LOGGING:
                print(f"[LLM_SAMPLING] Taking every {SAMPLE_SENTENCES_INTERVAL}th sentence:")
            sampled.append("\nEvery 100th sentence:")
            hundredthSentences = []
            for i in range(SAMPLE_SENTENCES_INTERVAL, len(sentences), SAMPLE_SENTENCES_INTERVAL):
                if ENABLE_DETAILED_LOGGING and SHOW_TEXT_PREVIEWS:
                    sentencePreview = sentences[i][:100] + "..." if len(sentences[i]) > 100 else sentences[i]
                    print(f"[LLM_SAMPLING]   {i+1}: '{sentencePreview}'")
                sampled.append(f"Sentence {i+1}: \"{sentences[i]}\"")
                hundredthSentences.append(i+1)
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[LLM_SAMPLING] Sampled sentences at positions: {hundredthSentences}")
        else:
            if ENABLE_DETAILED_LOGGING:
                print(f"[LLM_SAMPLING] Document too short for {SAMPLE_SENTENCES_INTERVAL}th sentence sampling")
        
        sampleText = "\n".join(sampled)
        if ENABLE_DETAILED_LOGGING:
            print(f"[LLM_SAMPLING] ✓ Sample text created, length: {len(sampleText)} characters")
        
        return sampleText
    
    def generateDocumentDescription(self, sentences, documentName):
        """Generate document description using LLM - helps users understand what the document is about"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[LLM_DESCRIPTION] Starting document description generation...")
            print(f"[LLM_DESCRIPTION] Document: {documentName}")
            print(f"[LLM_DESCRIPTION] Using model: {DESCRIPTION_MODEL}")
        
        try:
            sampleText = self.sampleSentencesForLlm(sentences)
            
            # Build the prompt using our template
            promptContent = PROMPTS['document_description']['user_template'].format(
                document_name=documentName,
                first_count=SAMPLE_SENTENCES_FIRST,
                sample_text=sampleText
            )
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[LLM_DESCRIPTION] Prompt length: {len(promptContent)} characters")
                if SHOW_TEXT_PREVIEWS:
                    print(f"[LLM_DESCRIPTION] Prompt preview:")
                    print(f"[LLM_DESCRIPTION] '{promptContent[:300]}...'")
            
            # Set up the conversation messages
            messages = [
                {'role': 'user', 'content': PROMPTS['document_description']['system']},
                {'role': 'assistant', 'content': PROMPTS['document_description']['assistant']},
                {'role': 'user', 'content': promptContent}
            ]
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[LLM_DESCRIPTION] Sending request to {DESCRIPTION_MODEL}...")
            startTime = time.time()
            response = ollama.chat(model=DESCRIPTION_MODEL, messages=messages)
            endTime = time.time()
            
            description = response['message']['content']
            if ENABLE_DETAILED_LOGGING:
                print(f"[LLM_DESCRIPTION] ✓ Response received in {endTime - startTime:.2f} seconds")
                print(f"[LLM_DESCRIPTION] Response length: {len(description)} characters")
                if SHOW_TEXT_PREVIEWS:
                    print(f"[LLM_DESCRIPTION] Description preview:")
                    print(f"[LLM_DESCRIPTION] '{description[:PREVIEW_LENGTH]}...'")
            
            return description
            
        except Exception as e:
            print(f"[LLM_DESCRIPTION] ✗ Error generating description with {DESCRIPTION_MODEL}: {e}")
            if ENABLE_DETAILED_LOGGING:
                traceback.print_exc()
            return f"Error generating description: {e}"
    
    def generateEmbedding(self, sentence, sentenceNum=None):
        """Generate embedding for a sentence - this creates the vector representation we use for similarity matching"""
        sentenceId = f"sentence {sentenceNum}" if sentenceNum else "sentence"
        if ENABLE_DETAILED_LOGGING:
            print(f"[EMBEDDING] Generating embedding for {sentenceId}...")
        
        try:
            if ENABLE_DETAILED_LOGGING and SHOW_TEXT_PREVIEWS:
                sentencePreview = sentence[:100] + "..." if len(sentence) > 100 else sentence
                print(f"[EMBEDDING] Text: '{sentencePreview}'")
                print(f"[EMBEDDING] Text length: {len(sentence)} characters")
                print(f"[EMBEDDING] Using model: {EMBEDDING_MODEL}")
            
            startTime = time.time()
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=sentence)
            endTime = time.time()
            
            embedding = response['embedding']
            if ENABLE_DETAILED_LOGGING:
                print(f"[EMBEDDING] ✓ Embedding generated in {endTime - startTime:.2f} seconds")
                print(f"[EMBEDDING] Embedding dimensions: {len(embedding)}")
                if SHOW_EMBEDDING_SAMPLES:
                    print(f"[EMBEDDING] Embedding sample (first {EMBEDDING_SAMPLE_SIZE} values): {embedding[:EMBEDDING_SAMPLE_SIZE]}")
            
            return embedding
        except Exception as e:
            print(f"[EMBEDDING] ✗ Error generating embedding for {sentenceId}: {e}")
            return None
    
    def calculateMetadata(self, sentences):
        """Calculate statistical metadata for the document - gives us insights into document structure"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[METADATA] Calculating document statistics...")
            print(f"[METADATA] Total sentences to analyze: {len(sentences)}")
        
        if not sentences:
            if ENABLE_DETAILED_LOGGING:
                print(f"[METADATA] ✗ No sentences provided")
            return {}
        
        # Calculate various statistics about sentence lengths and word counts
        sentenceLengths = [len(s) for s in sentences]
        wordCounts = [len(s.split()) for s in sentences]
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[METADATA] Sentence length range: {min(sentenceLengths)} - {max(sentenceLengths)} chars")
            print(f"[METADATA] Word count range: {min(wordCounts)} - {max(wordCounts)} words")
        
        metadata = {
            'totalSentences': len(sentences),
            'avgSentenceLength': statistics.mean(sentenceLengths),
            'stdSentenceLength': statistics.stdev(sentenceLengths) if len(sentences) > 1 else 0,
            'totalWordCount': sum(wordCounts),
            'avgWordsPerSentence': statistics.mean(wordCounts),
            'minSentenceLength': min(sentenceLengths),
            'maxSentenceLength': max(sentenceLengths),
            'medianSentenceLength': statistics.median(sentenceLengths)
        }
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[METADATA] ✓ Calculated metadata:")
            for key, value in metadata.items():
                print(f"[METADATA]   {key}: {value}")
        
        return metadata
    
    def saveSentenceFile(self, sentence, embedding, sentenceNum, outputDir, pageNum=None):
        """Save individual sentence file with embedding, metadata, and page number"""
        filename = f"sentence_{sentenceNum}.txt"
        filepath = outputDir / filename
        
        if ENABLE_DETAILED_LOGGING:
            print(f"[SAVE_SENTENCE] Saving {filename}...")
            print(f"[SAVE_SENTENCE] File path: {filepath}")
            print(f"[SAVE_SENTENCE] Sentence length: {len(sentence)} chars")
            print(f"[SAVE_SENTENCE] Has embedding: {embedding is not None}")
            print(f"[SAVE_SENTENCE] Page number: {pageNum if pageNum else 'Unknown'}")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write the sentence between markers so we can easily extract it later
                f.write("---sentence---\n")
                f.write(f"{sentence}\n")
                f.write("---sentence---\n")
                
                # Save the embedding as a comma-separated list
                if embedding:
                    f.write(f"Embedding: {', '.join(map(str, embedding))}\n")
                    if ENABLE_DETAILED_LOGGING:
                        print(f"[SAVE_SENTENCE] Wrote embedding with {len(embedding)} dimensions")
                else:
                    f.write("Embedding: Error generating embedding\n")
                    if ENABLE_DETAILED_LOGGING:
                        print(f"[SAVE_SENTENCE] ⚠ No embedding available")
                
                # Add metadata for easy reference
                f.write("Metadata:\n")
                f.write(f"Length: {len(sentence)} characters\n")
                f.write(f"Sentence Number: {sentenceNum}\n")
                f.write(f"Page Number: {pageNum if pageNum else 'Unknown'}\n")
                f.write(f"Word Count: {len(sentence.split())}\n")
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[SAVE_SENTENCE] ✓ Successfully saved {filename}")
            return True
        except Exception as e:
            print(f"[SAVE_SENTENCE] ✗ Error saving sentence file {filename}: {e}")
            return False
    
    def saveEmbeddingsNpy(self, embeddings, outputDir):
        """Save all embeddings as numpy array - this makes loading much faster later"""
        if ENABLE_DETAILED_LOGGING:
            print(f"[SAVE_EMBEDDINGS] Preparing to save embeddings array...")
            print(f"[SAVE_EMBEDDINGS] Total embeddings: {len(embeddings)}")
        
        try:
            # Filter out any None embeddings that failed to generate
            validEmbeddings = [emb for emb in embeddings if emb is not None]
            invalidCount = len(embeddings) - len(validEmbeddings)
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[SAVE_EMBEDDINGS] Valid embeddings: {len(validEmbeddings)}")
                print(f"[SAVE_EMBEDDINGS] Invalid embeddings: {invalidCount}")
            
            if validEmbeddings:
                embeddingsArray = np.array(validEmbeddings)
                filepath = outputDir / "embeddings.npy"
                
                if ENABLE_DETAILED_LOGGING:
                    print(f"[SAVE_EMBEDDINGS] Array shape: {embeddingsArray.shape}")
                    print(f"[SAVE_EMBEDDINGS] Array dtype: {embeddingsArray.dtype}")
                    print(f"[SAVE_EMBEDDINGS] Saving to: {filepath}")
                
                np.save(filepath, embeddingsArray)
                if ENABLE_DETAILED_LOGGING:
                    print(f"[SAVE_EMBEDDINGS] ✓ Saved {len(validEmbeddings)} embeddings to embeddings.npy")
                return True
            else:
                print(f"[SAVE_EMBEDDINGS] ✗ No valid embeddings to save")
                return False
        except Exception as e:
            print(f"[SAVE_EMBEDDINGS] ✗ Error saving embeddings: {e}")
            if ENABLE_DETAILED_LOGGING:
                traceback.print_exc()
            return False
    
    def saveMetadataFile(self, description, metadata, documentName, outputDir):
        """Save metadata file with description and statistics - gives users an overview of the document"""
        filepath = outputDir / "metadata.txt"
        if ENABLE_DETAILED_LOGGING:
            print(f"[SAVE_METADATA] Saving metadata file...")
            print(f"[SAVE_METADATA] File path: {filepath}")
            print(f"[SAVE_METADATA] Description length: {len(description)} chars")
            print(f"[SAVE_METADATA] Metadata entries: {len(metadata)}")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write the document description first
                f.write("Document Description:\n")
                f.write(f"{description}\n\n")
                
                # Then write all the statistics we calculated
                f.write("Document Statistics:\n")
                f.write(f"Document Name: {documentName}\n")
                f.write(f"Total Sentences: {metadata['totalSentences']}\n")
                f.write(f"Average Sentence Length: {metadata['avgSentenceLength']:.2f} characters\n")
                f.write(f"Standard Deviation of Sentence Lengths: {metadata['stdSentenceLength']:.2f} characters\n")
                f.write(f"Minimum Sentence Length: {metadata['minSentenceLength']} characters\n")
                f.write(f"Maximum Sentence Length: {metadata['maxSentenceLength']} characters\n")
                f.write(f"Median Sentence Length: {metadata['medianSentenceLength']:.2f} characters\n")
                f.write(f"Total Word Count: {metadata['totalWordCount']}\n")
                f.write(f"Average Words per Sentence: {metadata['avgWordsPerSentence']:.2f} words\n")
                
                # Add page information if we have it
                if self.documentPages:
                    f.write(f"Total Pages: {len(self.documentPages)}\n")
            
            if ENABLE_DETAILED_LOGGING:
                print(f"[SAVE_METADATA] ✓ Successfully saved metadata file")
            return True
        except Exception as e:
            print(f"[SAVE_METADATA] ✗ Error saving metadata file: {e}")
            return False
    
    def processDocument(self, filePath):
        """Process a single document - this is the main orchestration method"""
        print(f"\n{'='*80}")
        print(f"[PROCESS_DOC] STARTING DOCUMENT PROCESSING")
        print(f"[PROCESS_DOC] File: {filePath.name}")
        if ENABLE_DETAILED_LOGGING:
            print(f"[PROCESS_DOC] Full path: {filePath}")
            print(f"[PROCESS_DOC] File size: {filePath.stat().st_size} bytes")
        print(f"{'='*80}")
        
        # Reset page information for new document - clean slate for each document
        self.documentPages = None
        self.documentFullText = None
        
        # Step 1: Extract text based on file type
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 1: TEXT EXTRACTION")
        if filePath.suffix.lower() == '.pdf':
            if ENABLE_DETAILED_LOGGING:
                print(f"[PROCESS_DOC] Detected PDF file")
            text = self.extractTextFromPdf(filePath)
        elif filePath.suffix.lower() == '.txt':
            if ENABLE_DETAILED_LOGGING:
                print(f"[PROCESS_DOC] Detected TXT file")
            text = self.extractTextFromTxt(filePath)
        else:
            print(f"[PROCESS_DOC] ✗ Unsupported file type: {filePath.suffix}")
            return False
        
        if not text:
            print(f"[PROCESS_DOC] ✗ Failed to extract text from document")
            return False
        
        # Step 2: Clean the extracted text
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 2: TEXT CLEANING")
        text = self.cleanText(text)
        
        # Step 3: Split text into individual sentences
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 3: SENTENCE TOKENIZATION")
        sentences = self.splitIntoSentences(text)
        
        if not sentences:
            print(f"[PROCESS_DOC] ✗ No sentences found in document")
            return False
        
        # Step 4: Create output directory for this document
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 4: OUTPUT DIRECTORY SETUP")
        docName = filePath.stem
        outputDir = self.outputFolder / docName
        if ENABLE_DETAILED_LOGGING:
            print(f"[PROCESS_DOC] Creating output directory: {outputDir}")
        outputDir.mkdir(exist_ok=True)
        if ENABLE_DETAILED_LOGGING:
            print(f"[PROCESS_DOC] ✓ Output directory ready")
        
        # Step 5: Generate a nice description of what the document is about
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 5: DOCUMENT DESCRIPTION GENERATION")
        description = self.generateDocumentDescription(sentences, docName)
        
        # Step 6: Calculate statistical metadata
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 6: METADATA CALCULATION")
        metadata = self.calculateMetadata(sentences)
        
        # Step 7: Process each sentence individually - generate embeddings and find page numbers
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 7: SENTENCE PROCESSING, EMBEDDING GENERATION & PAGE DETECTION")
        print(f"[PROCESS_DOC] Processing {len(sentences)} sentences...")
        
        embeddings = []
        successfulSentences = 0
        failedSentences = 0
        pageDetectionStats = {'found': 0, 'notFound': 0}
        
        for i, sentence in enumerate(sentences, 1):
            if ENABLE_DETAILED_LOGGING:
                print(f"\n[PROCESS_DOC] Processing sentence {i}/{len(sentences)}")
            
            # Generate the vector embedding for this sentence
            embedding = self.generateEmbedding(sentence, i)
            embeddings.append(embedding)
            
            # Try to figure out which page this sentence came from
            pageNum = self.findSentencePageNumber(sentence)
            if pageNum:
                pageDetectionStats['found'] += 1
            else:
                pageDetectionStats['notFound'] += 1
            
            # Save the sentence file with all its information
            if self.saveSentenceFile(sentence, embedding, i, outputDir, pageNum):
                successfulSentences += 1
            else:
                failedSentences += 1
            
            # Show progress periodically so user knows we're still working
            if i % PROGRESS_UPDATE_INTERVAL == 0 or i == len(sentences):
                print(f"[PROCESS_DOC] Progress: {i}/{len(sentences)} sentences processed")
        
        print(f"\n[PROCESS_DOC] Sentence processing complete:")
        print(f"[PROCESS_DOC] ✓ Successful: {successfulSentences}")
        if failedSentences > 0:
            print(f"[PROCESS_DOC] ✗ Failed: {failedSentences}")
        print(f"[PROCESS_DOC] Page detection: {pageDetectionStats['found']} found, {pageDetectionStats['notFound']} not found")
        
        # Step 8: Save all embeddings as a single numpy array for fast loading
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 8: EMBEDDINGS ARRAY STORAGE")
        self.saveEmbeddingsNpy(embeddings, outputDir)
        
        # Step 9: Save the metadata and description
        if ENABLE_DETAILED_LOGGING:
            print(f"\n[PROCESS_DOC] STEP 9: METADATA STORAGE")
        self.saveMetadataFile(description, metadata, docName, outputDir)
        
        print(f"\n{'='*80}")
        print(f"[PROCESS_DOC] ✓ DOCUMENT PROCESSING COMPLETE!")
        print(f"[PROCESS_DOC] Output location: {outputDir}")
        print(f"[PROCESS_DOC] Files created:")
        print(f"[PROCESS_DOC]   - {successfulSentences} sentence files (with page numbers)")
        print(f"[PROCESS_DOC]   - 1 embeddings.npy file")
        print(f"[PROCESS_DOC]   - 1 metadata.txt file")
        if self.documentPages:
            print(f"[PROCESS_DOC] Page detection: {pageDetectionStats['found']}/{len(sentences)} sentences located")
        print(f"{'='*80}")
        
        return True


class PageNumberUpdater:
    """
    Standalone page number updater for existing processed documents
    Can be used to add page numbers to documents that were processed without this feature
    """
    def __init__(self, inputFolder='input', outputFolder='output'):
        self.inputFolder = Path(inputFolder)
        self.outputFolder = Path(outputFolder)
        self.debug = True
    
    def normalizeText(self, text):
        """Normalize text the same way as the main processor - consistency is crucial"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def extractPagesConsistently(self, pdfPath):
        """Extract text from each page using consistent normalization"""
        print(f"[EXTRACT] Processing: {pdfPath.name}")
        
        try:
            with open(pdfPath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pages = []
                fullTextParts = []
                
                # Extract each page with consistent processing
                for i, page in enumerate(reader.pages):
                    rawText = page.extract_text()
                    normalizedText = self.normalizeText(rawText)
                    
                    pages.append({
                        'pageNum': i + 1,
                        'text': normalizedText,
                        'rawText': rawText
                    })
                    
                    fullTextParts.append(normalizedText)
                    
                    if self.debug:
                        print(f"[EXTRACT] Page {i+1}: {len(normalizedText)} chars")
                
                fullText = ' '.join(fullTextParts)
                fullText = self.normalizeText(fullText)
                
                print(f"[EXTRACT] Total: {len(pages)} pages, {len(fullText)} chars")
                return pages, fullText
                
        except Exception as e:
            print(f"[EXTRACT] Error: {e}")
            return None, None
    
    def findSentenceInPages(self, sentence, pages, fullText):
        """Find which page contains a sentence using multiple strategies - same logic as main processor"""
        cleanSentence = self.normalizeText(sentence)
        
        if cleanSentence.endswith('.'):
            cleanSentence = cleanSentence[:-1]
        
        # Strategy 1: Direct text search - most reliable
        for pageInfo in pages:
            if cleanSentence in pageInfo['text']:
                if self.debug:
                    print(f"[MATCH] Direct match on page {pageInfo['pageNum']}")
                return pageInfo['pageNum']
        
        # Strategy 2: Fuzzy matching for OCR errors
        bestMatch = None
        bestRatio = 0.0
        threshold = 0.8
        
        for pageInfo in pages:
            if len(pageInfo['text']) < 50 or len(cleanSentence) > len(pageInfo['text']) * 0.8:
                continue
                
            ratio = SequenceMatcher(None, cleanSentence, pageInfo['text']).ratio()
            if ratio > bestRatio and ratio > threshold:
                bestRatio = ratio
                bestMatch = pageInfo['pageNum']
        
        if bestMatch:
            if self.debug:
                print(f"[MATCH] Fuzzy match on page {bestMatch} (ratio: {bestRatio:.3f})")
            return bestMatch
        
        # Strategy 3: Fragment matching - look for sentence pieces
        words = cleanSentence.split()
        if len(words) > 5:
            startFragment = ' '.join(words[:3])
            endFragment = ' '.join(words[-3:])
            
            for pageInfo in pages:
                if startFragment in pageInfo['text'] and endFragment in pageInfo['text']:
                    if self.debug:
                        print(f"[MATCH] Fragment match on page {pageInfo['pageNum']}")
                    return pageInfo['pageNum']
        
        # Strategy 4: Position-based estimation - last resort
        try:
            pos = fullText.find(cleanSentence)
            if pos != -1:
                charsPerPage = len(fullText) / len(pages)
                estimatedPage = min(int(pos / charsPerPage) + 1, len(pages))
                
                for pageOffset in [0, -1, 1, -2, 2]:
                    checkPage = estimatedPage + pageOffset
                    if 1 <= checkPage <= len(pages):
                        pageIdx = checkPage - 1
                        if cleanSentence in pages[pageIdx]['text']:
                            if self.debug:
                                print(f"[MATCH] Position verification found on page {checkPage}")
                            return checkPage
        except:
            pass
        
        if self.debug:
            print(f"[MATCH] No match found")
        return None
    
    def updateSentenceFile(self, sentenceFile, pageNum):
        """Update sentence file with page number information"""
        try:
            # Read existing content
            with open(sentenceFile, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove any existing page number lines
            lines = content.strip().split('\n')
            lines = [line for line in lines if not line.startswith("Page Number:")]
            
            # Add new page number
            pageInfo = str(pageNum) if pageNum else "Unknown"
            
            # Find where to insert it (before Word Count if it exists)
            insertPos = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("Word Count:"):
                    insertPos = i
                    break
            
            lines.insert(insertPos, f"Page Number: {pageInfo}")
            
            # Write back the updated content
            with open(sentenceFile, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            
            if self.debug:
                print(f"[UPDATE] ✓ {sentenceFile.name} → page {pageInfo}")
            return True
            
        except Exception as e:
            print(f"[UPDATE] ✗ Error updating {sentenceFile.name}: {e}")
            return False
    
    def processDocument(self, pdfPath):
        """Process a single document for page number updates"""
        print(f"\n{'='*60}")
        print(f"[PROCESS] {pdfPath.name}")
        print(f"{'='*60}")
        
        docName = pdfPath.stem
        outputDir = self.outputFolder / docName
        
        # Check if we have processed output for this document
        if not outputDir.exists():
            print(f"[PROCESS] ✗ No output folder: {outputDir}")
            return False
        
        # Extract pages from the PDF
        pages, fullText = self.extractPagesConsistently(pdfPath)
        if not pages:
            print(f"[PROCESS] ✗ Failed to extract pages")
            return False
        
        # Find all sentence files to update
        sentenceFiles = list(outputDir.glob("sentence_*.txt"))
        if not sentenceFiles:
            print(f"[PROCESS] ✗ No sentence files found")
            return False
        
        print(f"[PROCESS] Processing {len(sentenceFiles)} sentences...")
        
        stats = {'updated': 0, 'notFound': 0, 'errors': 0}
        
        # Update each sentence file
        for sentenceFile in sorted(sentenceFiles):
            try:
                # Read the sentence from the file
                with open(sentenceFile, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract the sentence text
                sentenceMatch = re.search(r'---sentence---\n(.*?)\n---sentence---', content, re.DOTALL)
                if not sentenceMatch:
                    print(f"[PROCESS] ✗ No sentence found in {sentenceFile.name}")
                    stats['errors'] += 1
                    continue
                
                sentence = sentenceMatch.group(1).strip()
                pageNum = self.findSentenceInPages(sentence, pages, fullText)
                
                # Update the file with the page number
                if self.updateSentenceFile(sentenceFile, pageNum):
                    if pageNum:
                        stats['updated'] += 1
                    else:
                        stats['notFound'] += 1
                else:
                    stats['errors'] += 1
                
            except Exception as e:
                print(f"[PROCESS] ✗ Error with {sentenceFile.name}: {e}")
                stats['errors'] += 1
        
        print(f"\n[RESULTS] Updated: {stats['updated']}, Not found: {stats['notFound']}, Errors: {stats['errors']}")
        return True
    
    def run(self):
        """Run the standalone page number updater"""
        print(f"[UPDATER] Starting page number update...")
        print(f"[UPDATER] Input: {self.inputFolder}")
        print(f"[UPDATER] Output: {self.outputFolder}")
        
        # Find PDF files to process
        pdfFiles = list(self.inputFolder.glob("*.pdf"))
        if not pdfFiles:
            print(f"[UPDATER] ✗ No PDF files found")
            return
        
        print(f"[UPDATER] Found {len(pdfFiles)} PDF files")
        
        successful = 0
        for pdfFile in pdfFiles:
            if self.processDocument(pdfFile):
                successful += 1
        
        print(f"\n{'='*60}")
        print(f"[UPDATER] ✓ COMPLETE: {successful}/{len(pdfFiles)} documents processed")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Use the enhanced DocumentProcessor for new documents
    processor = DocumentProcessor()