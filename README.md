# \# OmniBrain

# 

# \## Project Overview

# 

# OmniBrain is an intelligent document question-answering system designed to process documents, extract text and images, and route user queries to the appropriate AI agent.

# 

# The system uses a multi-agent architecture with LangGraph to handle different types of queries through Text, SQL, and Vision Agents.

# 

# \## Objectives

# 

# \- Extract and process information from PDF documents.

# \- Split extracted text into meaningful chunks for retrieval.

# \- Extract images from documents.

# \- Route user queries to the appropriate AI agent.

# \- Provide a FastAPI backend for query processing.

# \- Provide a Streamlit-based user interface.

# \- Integrate Self-RAG and Guardrails for improved reliability and safety.

# 

# \## System Architecture

# 

# ```text

# User

# &#x20; |

# &#x20; v

# Streamlit Interface

# &#x20; |

# &#x20; v

# FastAPI Backend

# &#x20; |

# &#x20; v

# LangGraph Supervisor

# &#x20; |

# &#x20; +------------------+------------------+

# &#x20; |                  |                  |

# &#x20; v                  v                  v

# Text Agent        SQL Agent        Vision Agent

# &#x20; |                  |                  |

# &#x20; +------------------+------------------+

# &#x20;                    |

# &#x20;                    v

# &#x20;                Response

