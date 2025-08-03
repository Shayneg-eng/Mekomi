"""
Main entry point for AI-Driven Document Analysis and Query Matching System
Provides menu interface for document processing and querying
"""

import traceback
from pathlib import Path

from config import SUPPORTED_EXTENSIONS, TARGET_SENTENCES, ENABLE_DETAILED_LOGGING

from document_processor import DocumentProcessor
from query_matcher import QueryMatcher

def checkDependencies():
    """Check if all required packages are available - prevents runtime errors later"""
    print("[STARTUP] Checking required packages...")
    try:
        import PyPDF2
        print("[STARTUP] ✓ PyPDF2 available")
        import ollama
        print("[STARTUP] ✓ ollama available")
        import numpy as np
        print("[STARTUP] ✓ numpy available")
        from sklearn.metrics.pairwise import cosine_similarity
        print("[STARTUP] ✓ scikit-learn available")
        return True
    except ImportError as e:
        print(f"[STARTUP] ✗ Missing required package: {e}")
        print("[STARTUP] Please install required packages:")
        print("[STARTUP] pip install PyPDF2 ollama numpy scikit-learn")
        return False

def processDocumentsMenu(processor):
    """Handle document processing menu option - processes all files in the input folder"""
    print(f"\n[MENU] OPTION 1: DOCUMENT PROCESSING")
    print("="*60)
    
    # Find all supported files in the input folder
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(list(processor.inputFolder.glob(f"*{ext}")))
    
    if not files:
        print(f"[MENU] ✗ No supported files found in {processor.inputFolder}")
        print(f"[MENU] Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        print(f"[MENU] Please add some documents to process.")
        return
    
    # Show user what we found
    print(f"[MENU] Found {len(files)} files to process:")
    for i, file in enumerate(files, 1):
        print(f"[MENU]   {i}. {file.name} ({file.stat().st_size} bytes)")
    
    input(f"\n[MENU] Press Enter to continue with processing...")
    
    # Process each file
    successful = 0
    for i, filePath in enumerate(files, 1):
        print(f"\n[MENU] Processing file {i}/{len(files)}: {filePath.name}")
        try:
            if processor.processDocument(filePath):
                successful += 1
                print(f"[MENU] ✓ Successfully processed {filePath.name}")
            else:
                print(f"[MENU] ✗ Failed to process {filePath.name}")
        except Exception as e:
            print(f"[MENU] ✗ Error processing {filePath.name}: {e}")
            if ENABLE_DETAILED_LOGGING:
                traceback.print_exc()
    
    # Show final results
    print(f"\n[MENU] PROCESSING SUMMARY:")
    print(f"[MENU] Total files: {len(files)}")
    print(f"[MENU] Successfully processed: {successful}")
    print(f"[MENU] Failed: {len(files) - successful}")

def queryDocumentsMenu(matcher):
    """Handle document querying menu option - searches across all processed documents"""
    print(f"\n[MENU] OPTION 2: MULTI-DOCUMENT QUERYING")
    print("="*60)
    
    # Check what documents are available for querying
    availableDocs = matcher.getAvailableDocuments()
    
    if not availableDocs:
        print(f"[MENU] ✗ No processed documents found. Please process some documents first.")
        return
    
    # Show user what documents are available
    print(f"[MENU] Available documents ({len(availableDocs)}):")
    for i, doc in enumerate(availableDocs, 1):
        docPath = matcher.outputFolder / doc
        sentenceCount = len([f for f in docPath.iterdir() if f.name.startswith("sentence_")])
        print(f"[MENU]   {i}. {doc} ({sentenceCount} sentences)")
    
    print(f"\n[MENU] This query will search across ALL {len(availableDocs)} documents simultaneously.")
    
    # Get the user's question
    userQuery = input("\n[MENU] Enter your question: ").strip()
    print(f"[MENU] User query: '{userQuery}'")
    
    if userQuery:
        # Ask how many relevant sentences they want
        print(f"\n[MENU] How many relevant sentences would you like to find?")
        print(f"[MENU] (Default: {TARGET_SENTENCES} sentences)")
        
        targetInput = input(f"[MENU] Enter target sentences (press Enter for default): ").strip()
        
        if targetInput:
            try:
                targetSentences = int(targetInput)
                if targetSentences <= 0:
                    print(f"[MENU] ⚠ Invalid number. Using default: {TARGET_SENTENCES}")
                    targetSentences = TARGET_SENTENCES
                else:
                    print(f"[MENU] Target sentences set to: {targetSentences}")
            except ValueError:
                print(f"[MENU] ⚠ Invalid input. Using default: {TARGET_SENTENCES}")
                targetSentences = TARGET_SENTENCES
        else:
            targetSentences = TARGET_SENTENCES
            print(f"[MENU] Using default target sentences: {targetSentences}")
        
        # Execute the query
        matcher.queryAllDocuments(userQuery, targetSentences)
    else:
        print("[MENU] ✗ No query entered.")

def main():
    """Main function with menu system - the central control point for the application"""
    print("AI-Driven Document Analysis and Query Matching System")
    print("Enhanced Multi-Document Search with Dynamic Similarity Threshold")
    print("=" * 80)
    
    # Make sure all required packages are installed
    if not checkDependencies():
        return
    
    print("[STARTUP] ✓ All required packages available")
    
    # Initialize our main components
    processor = DocumentProcessor()
    matcher = QueryMatcher()
    
    # Main application loop
    while True:
        print(f"\n{'='*60}")
        print("MAIN MENU")
        print("="*60)
        print("1. Process documents")
        print("2. Ask a question (searches all documents)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        if ENABLE_DETAILED_LOGGING:
            print(f"[MENU] User selected option: {choice}")
        
        if choice == "1":
            processDocumentsMenu(processor)
        
        elif choice == "2":
            queryDocumentsMenu(matcher)
        
        elif choice == "3":
            print("[MENU] Goodbye!")
            break
        
        else:
            print(f"[MENU] ✗ Invalid choice: {choice}. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()