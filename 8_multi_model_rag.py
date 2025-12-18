{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "12bea030",
   "metadata": {},
   "source": [
    "# System Dependencies\n",
    "\n",
    "To get started with Unstructured.io, we need a few system-wide dependencies: \n",
    "\n",
    "## Poppler (poppler-utils)\n",
    "Handles PDF processing. It's a library that can extract text, images, and metadata from PDFs. Unstructured uses it to parse PDF documents and convert them into processable text.\n",
    "\n",
    "## Tesseract (tesseract-ocr) \n",
    "Optical Character Recognition (OCR) engine. When you have scanned documents, images with text, or PDFs that are essentially pictures, Tesseract reads the text from these images and converts it to machine-readable text.\n",
    "\n",
    "## libmagic\n",
    "File type detection library. It identifies what type of file you're dealing with (PDF, Word doc, image, etc.) by analyzing the file's content, not just the extension. This helps Unstructured choose the right processing method for each document."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "335f25b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# for linux\n",
    "# !apt-get install poppler-utils tesseract-ocr libmagic-dev\n",
    "\n",
    "# for mac\n",
    "# !brew install poppler tesseract libmagic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fddda27d",
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install -Uq \"unstructured[all-docs]\" \n",
    "%pip install -Uq langchain_chroma \n",
    "%pip install -Uq langchain langchain-community langchain-openai \n",
    "%pip install -Uq python_dotenv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "144884d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "from typing import List\n",
    "\n",
    "# Unstructured for document parsing\n",
    "from unstructured.partition.pdf import partition_pdf\n",
    "from unstructured.chunking.title import chunk_by_title\n",
    "\n",
    "# LangChain components\n",
    "from langchain_core.documents import Document\n",
    "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n",
    "from langchain_chroma import Chroma\n",
    "from langchain_core.messages import HumanMessage\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "load_dotenv()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78fb6efb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def partition_document(file_path: str):\n",
    "    \"\"\"Extract elements from PDF using unstructured\"\"\"\n",
    "    print(f\"📄 Partitioning document: {file_path}\")\n",
    "    \n",
    "    elements = partition_pdf(\n",
    "        filename=file_path,  # Path to your PDF file\n",
    "        strategy=\"hi_res\", # Use the most accurate (but slower) processing method of extraction\n",
    "        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text\n",
    "        extract_image_block_types=[\"Image\"], # Grab images found in the PDF\n",
    "        extract_image_block_to_payload=True # Store images as base64 data you can actually use\n",
    "    )\n",
    "    \n",
    "    print(f\"✅ Extracted {len(elements)} elements\")\n",
    "    return elements\n",
    "\n",
    "# Test with your PDF file\n",
    "file_path = \"./docs/attention-is-all-you-need.pdf\"  # Change this to your PDF path\n",
    "elements = partition_document(file_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2524980c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# elements\n",
    "# len(elements)\n",
    "\n",
    "\n",
    "elements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "802f4cdd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# All types of different atomic elements we see from unstructured\n",
    "set([str(type(el)) for el in elements])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9b3cd26",
   "metadata": {},
   "outputs": [],
   "source": [
    "elements[36].to_dict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7abe8c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Gather all images\n",
    "images = [element for element in elements if element.category == 'Image']\n",
    "print(f\"Found {len(images)} images\")\n",
    "\n",
    "images[0].to_dict()\n",
    "\n",
    "# Use https://codebeautify.org/base64-to-image-converter to view the base64 text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64d9f175",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Gather all table\n",
    "tables = [element for element in elements if element.category == 'Table']\n",
    "print(f\"Found {len(tables)} tables\")\n",
    "\n",
    "tables[0].to_dict()\n",
    "\n",
    "# Use https://jsfiddle.net/ to view the table html \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ecef416",
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_chunks_by_title(elements):\n",
    "    \"\"\"Create intelligent chunks using title-based strategy\"\"\"\n",
    "    print(\"🔨 Creating smart chunks...\")\n",
    "    \n",
    "    chunks = chunk_by_title(\n",
    "        elements, # The parsed PDF elements from previous step\n",
    "        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk\n",
    "        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters\n",
    "        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors\n",
    "    )\n",
    "    \n",
    "    print(f\"✅ Created {len(chunks)} chunks\")\n",
    "    return chunks\n",
    "\n",
    "# Create chunks\n",
    "chunks = create_chunks_by_title(elements)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94189379",
   "metadata": {},
   "outputs": [],
   "source": [
    "# View all chunks\n",
    "# chunks\n",
    "\n",
    "# All unique types\n",
    "set([str(type(chunk)) for chunk in chunks])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6410fa1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# View a single chunk\n",
    "# chunks[2].to_dict()\n",
    "\n",
    "# View original elements\n",
    "chunks[11].metadata.orig_elements[-1].to_dict()\n",
    "# Note: 4th chunk has the first image + 11th chunk has the first table in the sample PDF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0967ab4c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def separate_content_types(chunk):\n",
    "    \"\"\"Analyze what types of content are in a chunk\"\"\"\n",
    "    content_data = {\n",
    "        'text': chunk.text,\n",
    "        'tables': [],\n",
    "        'images': [],\n",
    "        'types': ['text']\n",
    "    }\n",
    "    \n",
    "    # Check for tables and images in original elements\n",
    "    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):\n",
    "        for element in chunk.metadata.orig_elements:\n",
    "            element_type = type(element).__name__\n",
    "            \n",
    "            # Handle tables\n",
    "            if element_type == 'Table':\n",
    "                content_data['types'].append('table')\n",
    "                table_html = getattr(element.metadata, 'text_as_html', element.text)\n",
    "                content_data['tables'].append(table_html)\n",
    "            \n",
    "            # Handle images\n",
    "            elif element_type == 'Image':\n",
    "                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):\n",
    "                    content_data['types'].append('image')\n",
    "                    content_data['images'].append(element.metadata.image_base64)\n",
    "    \n",
    "    content_data['types'] = list(set(content_data['types']))\n",
    "    return content_data\n",
    "\n",
    "def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:\n",
    "    \"\"\"Create AI-enhanced summary for mixed content\"\"\"\n",
    "    \n",
    "    try:\n",
    "        # Initialize LLM (needs vision model for images)\n",
    "        llm = ChatOpenAI(model=\"gpt-4o\", temperature=0)\n",
    "        \n",
    "        # Build the text prompt\n",
    "        prompt_text = f\"\"\"You are creating a searchable description for document content retrieval.\n",
    "\n",
    "        CONTENT TO ANALYZE:\n",
    "        TEXT CONTENT:\n",
    "        {text}\n",
    "\n",
    "        \"\"\"\n",
    "        \n",
    "        # Add tables if present\n",
    "        if tables:\n",
