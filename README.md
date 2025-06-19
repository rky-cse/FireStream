```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Input    │     │ Context Signals │     │   Watch Events  │     │   Chat/Social   │
│ (Voice, Search) │     │(Weather, Time)  │     │(Play, Complete) │     │   Interactions  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Ingestion Layer                                      │
│                                                                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ Voice/Text  │    │  Context    │    │ Playback    │    │ Chat Sentiment          │  │
│  │ Processing  │    │  Fetchers   │    │ Tracker     │    │ Analysis                │  │
│  │(Whisper+STT)│    │(Time/Weather│    │(Events &    │    │(HuggingFace             │  │
│  │(openSMILE)  │    │ /Festivals) │    │ Completion) │    │ Sentiment Model)        │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └───────────┬─────────────┘  │
└─────────┼──────────────────┼──────────────────┼───────────────────────┼────────────────┘
          │                  │                  │                       │
          ▼                  ▼                  ▼                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Feature Store Layer                                    │
│                                                                                        │
│  ┌───────────────────────────────────┐          ┌───────────────────────────────────┐  │
│  │ Real-time Features (Redis)        │          │ Historical Features (PostgreSQL)  │  │
│  │ - Current mood/sentiment vectors  │          │ - User profiles                   │  │
│  │ - Live group emotions             │          │ - Watch history                   │  │
│  │ - Contextual signals (weather)    │          │ - Content metadata & scene tags   │  │
│  │ - Session state                   │          │ - Social connections              │  │
│  └────────────────┬──────────────────┘          └──────────────────┬────────────────┘  │
└───────────────────┼────────────────────────────────────────────────┼───────────────────┘
                    │                                                │
                    ▼                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Recommendation Engine Layer                               │
│                                                                                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ User Embeddings │    │ Content         │    │ LightFM Hybrid  │    │ Social       │ │
│  │ Generator       │    │ Embeddings      │    │ Recommender     │    │ Signal       │ │
│  │ - Watch patterns│    │ - Scene tags    │    │ - Base rankings │    │ Processor    │ │
│  │ - Time patterns │    │ - Genre vectors │    │                 │    │              │ │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘    └──────┬───────┘ │
│           │                      │                      │                     │        │
│           └──────────────────────┼──────────────────────┼─────────────────────┘        │
│                                  │                      │                              │
│                                  ▼                      ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           Contextual Reranker                                   │   │
│  │ - Time-aware boosting                                                           │   │
│  │ - Weather-based adjustments                                                     │   │
│  │ - Festival/holiday significance                                                 │   │
│  │ - Group dynamics handling                                                       │   │
│  │ - Friend activity integration                                                   │   │
│  └───────────────────────────────────────┬─────────────────────────────────────────┘   │
└──────────────────────────────────────────┼─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   API Layer                                            │
│                                                                                        │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ Recommendation API  │    │ Social Notification │    │ Explanation Generator       │ │
│  │ (FastAPI)           │    │ API                 │    │ (Reason formatting)         │ │
│  └─────────────┬───────┘    └──────────┬──────────┘    └─────────────┬───────────────┘ │
└────────────────┼───────────────────────┼─────────────────────────────┼─────────────────┘
                 │                       │                             │
                 ▼                       ▼                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Client Layer                                          │
│                                                                                        │
│  ┌───────────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐   │
│  │ FireTV App UI             │    │ Group Watch Mode     │    │ Social Overlays    │   │
│  │ - Recommendations         │    │ - Shared controls    │    │ - Friend activity  │   │
│  │ - Content browsing        │    │ - Group sentiment    │    │ - Chat integration │   │
│  │ - Voice interactions      │    │ - Aggregate feedback │    │ - E-commerce links │   │
│  └───────────────────────────┘    └──────────────────────┘    └────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
