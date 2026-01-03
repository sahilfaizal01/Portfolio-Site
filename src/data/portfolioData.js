export const portfolioData = {
  profile: {
    name: "Sahil Faizal",
    title: "AI Technical Program Manager",
    tagline: "Building scalable AI systems and leading technical teams",
    email: "sahilfaiz2025@gmail.com",
    linkedin: "https://linkedin.com/in/sahil-faizal",
    github: "https://github.com/sahilfaizal01",
    twitter: "https://x.com/sahilfaiz074",
  },

  skills: [
    {
      category: "AI/ML",
      items: [
        "Post-Training Pipelines",
        "Fine-Tuning & Adaptation",
        "Computer Vision",
        "RAG Systems & Evaluation",
        "Reinforcement Learning",
        "LLM Optimization",
      ],
    },
    {
      category: "Leadership",
      items: [
        "Team Leadership",
        "AI Product Strategy",
        "Architecture Reviews",
        "Cross-Functional Execution",
        "Stakeholder Management",
        "Roadmap Planning",
      ],
    },
    {
      category: "Engineering",
      items: [
        "Python",
        "C++",
        "Data Science",
        "Big Data",
        "Distributed Systems",
        "Cloud Architecture",
        "GPU Programming",
        "DevOps",
        "Backend Dev",
      ],
    },
    {
      category: "Specializations",
      items: [
        "AI Inference",
        "Model Training at Scale",
        "MLOps",
        "Agentic Systems",
        "Performance Optimization",
        "GPU Kernel Tuning",
        "AI Infrastructure and Platforms",
      ],
    },
  ],

  achievements: [
    "Led the development of 8 critical LLM optimization features in 10 weeks, achieving a 60% reduction in model execution time",
    "Defined and streamlined the technical execution plan to deliver a custom LLM serving backend in a 3× shorter timeline",
    "Orchestrated MLPerf Inference v5.0 submission for SDXL, achieving 97% of the target performance.",
    "Published 6+ papers in reputed AI journals and conferences",
    "Recipient of the Graduate Scholarship – New York University, USA",
    "Excellence Award – Samsung R&D Institute, India",
    "Raman Research Award – Vellore Institute of Technology, India",
    "MITACS Globalink Research Award – Dalhousie University, Canada",
    "Winner, New Age Entrepreneurs Ideathon – organized by PALS, IIT Madras, India",
    "Most Nominated Winner, Arch-a-thon – School of Computer Science, VIT, India",
  ],

  experiences: [
    {
      company: "Advanced Micro Devices (AMD)",
      position: "AI Technical Program Manager - AI Platforms",
      location: "San Jose, CA",
      startDate: "2025-06",
      endDate: "Present",
      description:
        "Led end-to-end development, optimization, and delivery of AI inference software and infrastructure on high-performance GPU accelerators, driving low-latency, scalable solutions for LLMs, vision models, and ROCm platforms.",
      highlights: [
        "Enabled support for custom Python backends on AMD GPUs and implemented transformer-based models with dynamic batching as representative workloads.",
        "Designed and developed the inference engine, profiling workloads, analyzing bottlenecks, and coordinating across 4+ teams to deliver efficient, scalable LLM/VLM inference, including MLPerf targets",
        "Extended ROCm capabilities on Radeon/Navi GPUs and Ryzen APUs, improving test coverage, infrastructure deployment, and CI validation to ensure robust, secure, and high-performance GPU software delivery.",
      ],
    },
    {
      company: "Advanced Micro Devices (AMD)",
      position: "AI Technical Program Manager Intern - AI Infrastructure",
      location: "Santa Clara, CA",
      startDate: "2025-01",
      endDate: "2025-05",
      description:
        "Led QA, infrastructure improvements, and release management for AMD’s SHARK-AI inference engine and open-source ML CI/CD programs, enhancing scalability, efficiency, and cross-functional collaboration.",
      highlights: [
        "Conducted QA testing as the primary user of SHARK-AI, running inference on models like SDXL, Flux, Mistral, and Llama on AMD GPUs.",
        "Orchestrated customer onboarding to clusters and improved infrastructure efficiency through capacity planning, Kubernetes configuration, and scalability support.",
        "Prepared release documentation, facilitated cross-functional collaboration, and delivered executive updates to communicate project status, risks, and wins.",
        "Led status meetings to identify and resolve technical bottlenecks, improving project velocity and supporting large-scale AI workloads.",
      ],
    },
    {
      company: "New York University (NYU)",
      position: "Data Analyst",
      location: "New York, NY",
      startDate: "2024-01",
      endDate: "2025-05",
      description:
        "Engineered ETL pipelines, performed data analytics, and automation solutions to deliver actionable insights and AI-powered solutions for operational use.",
      highlights: [
        "Reduced data delivery time via optimized ETL pipelines and automated ingestion.",
        "Delivered actionable insights from large datasets, generating savings through predictive analysis.",
        "Built a text document processing software to retrieve and rank information from a database, improving information access and query efficiency.",
      ],
    },

    {
      company: "New York University (NYU)",
      position: "DevOps Intern",
      location: "New York, NY",
      startDate: "2024-09",
      endDate: "2024-12",
      description:
        "Built Kubernetes-native security and observability tools to strengthen platform security, monitor vulnerabilities, and provide actionable dashboards for HPC clusters.",
      highlights: [
        "Developed a dashboard identifying 5000+ weak passwords, improving HPC cluster security.",
        "Engineered a continuous security scanning solution detecting critical container vulnerabilities.",
        "Integrated Kubernetes API tools with Grafana/Prometheus to enhance platform observability and user adoption.",
      ],
    },

    {
      company: "Johnson & Johnson (J&J)",
      position: "Computer Vision Intern",
      location: "Redwood City, CA",
      startDate: "2024-05",
      endDate: "2024-08",
      description:
        "Designed and deployed a real-time segmentation algorithm and scalable MLOps pipelines for surgical video analysis to improve surgical outcomes.",

      highlights: [
        "Developed segmentation algorithms to accurately localize surgical targets while reducing false positives.",
        "Implemented MLOps pipelines for efficient experiment tracking, model retraining, and distributed inference on large-scale surgical video datasets.",
        "Conducted comparative performance analysis by fine-tuning vision foundation models, identifying the best architectures for surgical video understanding.",
        "Designed custom performance metrics and modules to quantify and enhance model robustness, supporting faster and safer surgical analysis.",
      ],
    },

    {
      company: "NYU Langone Health",
      position: "Research Associate",
      location: "New York, NY",
      startDate: "2023-06",
      endDate: "2024-05",
      description:
        "Developed AI-powered scene-specific semantic segmentation models to assist persons with blindness or low vision (pBLV), improving object localization, low-light resilience, and daily navigation independence.",
      highlights: [
        "Built scene-specific segmentation models for indoor spaces, outperforming generic models in object identification and resilience to low-light conditions.",
        "Collected, processed, and analyzed large-scale image datasets, fine-tuning deep learning architectures to optimize real-world assistive performance.",
        "Contributed to a paper demonstrating enhanced semantic segmentation, improving functional independence and supporting mobile deployment strategies.",
      ],
    },

    {
      company: "Nanyang Technological University (NTU)",
      position: "Research Engineer Intern",
      location: "Remote",
      startDate: "2022-08",
      endDate: "2023-08",
      description:
        "Built AI-driven GUI software integrating deep learning for classroom emotion analysis, enabling educators to extract actionable insights from video data.",
      highlights: [
        "Delivered a 75% improvement in actionable insights with PyQt5-based GUI for academic emotion analytics.",
        "Achieved 80% landmark tracking success with CNN-based facial recognition for 35% faster analysis.",
        "Improved academic emotion detection accuracy by 60% through multi-feature analytics integration.",
      ],
    },

    {
      company: "MITACS",
      position: "Globalink Research Intern",
      location: "Nova Scotia, Canada",
      startDate: "2022-05",
      endDate: "2022-08",
      description:
        "Developed computer vision systems for automated agricultural analysis, deploying deep learning solutions on embedded platforms for high-throughput sorting and anomaly detection.",
      highlights: [
        "Built a potato sorting system with 90% size estimation precision under challenging conditions.",
        "Deployed deep learning on embedded systems achieving 30 FPS processing and 93% anomaly detection precision.",
        "Conducted cluster analysis achieving 0.9 ARI for reliable tuber dimension classification.",
      ],
    },

    {
      company: "Samsung R&D Institute",
      position: "PRISM R&D Intern",
      location: "Bangalore, India",
      startDate: "2021-11",
      endDate: "2022-08",
      description:
        "Created full-stack AI applications combining NLP and generative models for music composition and video context synthesis, optimizing user experience on cloud platforms.",
      highlights: [
        "Fine-tuned NLP models for music composition with 50ms inference time, enabling real-time generation.",
        "Built a full-stack web application for context-aware music-video synthesis, reducing load time by 25%.",
        "Integrated conditional generative models with LSTM-based video classifiers to enhance content personalization.",
      ],
    },
  ],

  projects: [
    {
      title: "LLM Inference Optimizer",
      description:
        "Open-source toolkit for optimizing large language model inference performance",
      technologies: ["Python", "PyTorch", "CUDA", "Docker"],
      github: "https://github.com/yourprofile/llm-optimizer",
      demo: "https://demo.example.com",
      highlights: [
        "Achieved 3x speedup in inference latency",
        "Support for multiple model architectures",
        "1000+ GitHub stars",
      ],
    },
    {
      title: "AI Training Pipeline",
      description:
        "Distributed training framework for computer vision models at scale",
      technologies: ["Python", "TensorFlow", "Kubernetes", "Ray"],
      github: "https://github.com/yourprofile/training-pipeline",
      highlights: [
        "Scaled training to 100+ GPUs",
        "Automated hyperparameter tuning",
        "Reduced training time by 50%",
      ],
    },
    {
      title: "MLOps Dashboard",
      description:
        "Real-time monitoring and observability platform for ML models in production",
      technologies: ["React", "Python", "PostgreSQL", "Grafana"],
      demo: "https://mlops-demo.example.com",
      highlights: [
        "Tracks 100+ models in production",
        "Real-time performance metrics",
        "Automated alerting system",
      ],
    },
  ],

  publications: [
    {
      title:
        "Training Indoor and Scene-Specific Semantic Segmentation Models to Assist Blind and Low Vision Users in Activities of Daily Living",
      authors: [
        "Ruijie Sun",
        "Giles Hamilton-Fletcher",
        "Sahil Faizal",
        "Chen Feng",
        "Todd E Hudson",
        "John-Ross Rizzo",
        "Kevin C Chan",
      ],
      venue: "IEEE Open Journal of Engineering in Medicine and Biology",
      year: "2025",
      type: "Journal",
      location: "USA",
      pdf: "https://ieeexplore.ieee.org/abstract/document/11153825",
      abstract:
        "Goal: Persons with blindness or low vision (pBLV) face challenges in completing activities of daily living (ADLs/IADLs). Semantic segmentation techniques on smartphones, like DeepLabV3+, can quickly assist in identifying key objects, but their performance across different indoor settings and lighting conditions remains unclear. Methods: Using the MIT ADE20K SceneParse150 dataset, we trained and evaluated AI models for specific indoor scenes (kitchen, bedroom, bathroom, living room) and compared them with a generic indoor model. Performance was assessed using mean accuracy and intersection-over-union metrics. Results: Scene-specific models outperformed the generic model, particularly in identifying ADL/IADL objects. Models focusing on rooms with more unique objects showed the greatest improvements (bedroom, bathroom). Scene-specific models were also more resilient to low-light conditions. Conclusions: These findings highlight how using scene-specific models can boost key performance indicators for assisting pBLV across different functional environments. We suggest that a dynamic selection of the best-performing models on mobile technologies may better facilitate ADLs/IADLs for pBLV.",
    },
    {
      title:
        "EmoRoom: Unveiling Academic Emotions Through Interactive Visual Analytics in Classroom Videos",
      authors: [
        "Rajamanickam Yuvaraj",
        "Sahil Faizal",
        "Rajalakshmi Ratnavel",
        "Wang Yang",
      ],
      venue: "Future Technologies Conference (FTC) 2024",
      year: "2024",
      type: "Conference",
      location: "London, UK",
      pdf: "https://link.springer.com/chapter/10.1007/978-3-031-73122-8_39",
      abstract:
        "Measuring emotions in educational settings can provide important information in predicting and explaining student learning outcomes. Knowledge of student’s classroom emotions can also help teachers understand their students’ learning behaviors, improve their teaching methods, and optimize students’ learning and development. However, it can be highly challenging for teachers to monitor and accurately understand student emotions within classroom or group contexts, especially while they are actively teaching and attending to many students simultaneously. Video recording classroom activity can address that issue as high-definition cameras can be used to continuously record groups of students, enabling online or offline analyses of student emotions and supplementing teacher monitoring, retrieval, and interpretation of those processes. Teachers find it difficult to use existing emotion recognition methods to analyze student behaviors in videos due to a lack of user-friendly interfaces that facilitate automatic analysis. To address this challenge, we developed EmoRoom, an open-source tool designed to simplify the annotation and analysis of videos from an emotional perspective. EmoRoom tool offers a practical solution for quantifying and visualizing the frequency of emotions for individuals of interest. Furthermore, it can assist teachers in refining their teaching methods by offering a comprehensive overview of students’ emotional experiences during learning.",
    },
    {
      title:
        "RiceSeedNet: Rice seed variety identification using deep neural network",
      authors: [
        "Ratnavel Rajalakshmi",
        "Sahil Faizal",
        "S. Sivasankaran",
        "R. Geetha",
      ],
      venue: "Journal of Agriculture and Food Research",
      year: "2024",
      type: "Journal",
      pdf: "hhttps://www.sciencedirect.com/science/article/pii/S2666154324000991",
      abstract:
        "Rice is one of the most important food crops in the South India. Many varieties of rice are cultivated in different regions of the India to meet the dietary needs of the ever-growing population. In spite of huge investment in terms of land, labour, raw materials and machinery, the farmers continuously face irrecoverable loss due to various reasons like climatic changes, drought situation and seed quality. In the current practice, the quality of the seeds is certified by the Seed Testing Laboratories (STL) and purity analysis is done manually by trained technicians. However, seed classification is not uniform across different labs, due to several factors like fatigue, eye-strain and personal circumstances of the technicians. Hence, automated rice seed variety identification becomes a crucial task for ensuring the quality and germination potential of rice crops. This research is focused on the application of Deep Neural Network (RiceSeedNet) combined with traditional image processing techniques to classify local rice seed varieties of southern Tamilnadu, India. The RiceSeed Image corpus is created for this purpose considering 13 local varieties. The captured RGB images of rice seed data consists of 13,000 images of local rice seed varieties, having 1000 images for each variety. To automate the rice seed varietal identification, vision transformer-based architecture RiceSeedNet is developed. The proposed RiceSeedNet is 97% accurate in classifying the 13 local varieties of rice seeds. The RiceSeedNet was also evaluated on a publicly available rice grain data set to study the performance of the proposed model across the different rice grain varieties. On this cross-data validation, RiceSeedNet is able to achieve 99% accuracy in classifying 8 varieties of rice grains on the public dataset.",
    },

    {
      title:
        "Automated cataract disease detection on anterior segment eye images using adaptive thresholding and fine tuned inception-v3 model",
      authors: [
        "Sahil Faizal",
        "Charu Anant Rajput",
        "Rupali Tripathi",
        "Bhumika Verma",
        "Manas Ranjan Prusty",
        "Shivani Sachin Korade",
      ],
      venue: "Biomedical Signal Processing and Control",
      year: "2023",
      type: "Journal",
      pdf: "https://www.sciencedirect.com/science/article/abs/pii/S1746809422010047",
      abstract:
        "Early detection of cataracts plays a vital role in ensuring the prevention of vision loss. This paper aims to propose an algorithm that will act as an assistive measure in the process of automating cataract disease detection. The majority of the existing works are focused on the utilization of either fundus images, slit lamp images, or visible wavelength images captured using a DSLR camera. The novelty of this proposed algorithm is the capability to deliver equally accurate and precise performance using both normally captured visible wavelength images as well as medically captured anterior segment images which can, in turn, prove to be cost-effective as well. The image pre-processing techniques particularly adaptive thresholding hereby employed provides fast and accurate results on the input dataset fed to the CNN model which in turn is a fine-tuned version of Inception-v3. Here the training of the model has been done using visible wavelength images whereas the validation testing has been done using the anterior segment eye images medically obtained from a hospital. The proposed image pre-processing technique along with the model architecture ensures the achievement of a high classification accuracy of about 95%. Since the model deals with the examination of the anterior segment of the image, cases concerning nuclear cataract, cortical cataract, or a hybrid case involving the detection of both the aforementioned cataract types lie under the range of detection of our system.",
    },

    {
      title:
        "Automated Identification of Tree Species by Bark Texture Classification Using Convolutional Neural Networks",
      authors: ["Sahil Faizal"],
      venue:
        "International Journal for Research in Applied Science & Engineering Technology (IJRASET)",
      year: "2022",
      type: "Journal",
      pdf: "https://arxiv.org/abs/2210.09290",
      abstract:
        "Identification of tree species plays a key role in forestry related tasks like forest conservation, disease diagnosis and plant production. There had been a debate regarding the part of the tree to be used for differentiation, whether it should be leaves, fruits, flowers or bark. Studies have proven that bark is of utmost importance as it will be present despite seasonal variations and provides a characteristic identity to a tree by variations in the structure. In this paper, a deep learning based approach is presented by leveraging the method of computer vision to classify 50 tree species, on the basis of bark texture using the BarkVN-50 dataset. This is the maximum number of trees being considered for bark classification till now. A convolutional neural network(CNN), ResNet101 has been implemented using transfer-learning based technique of fine tuning to maximise the model performance. The model produced an overall accuracy of >94% during the evaluation. The performance validation has been done using K-Fold Cross Validation and by testing on unseen data collected from the Internet, this proved the model's generalization capability for real-world uses.",
    },

    {
      title: "Wild Animal Classifier Using CNN",
      authors: ["Sahil Faizal", "Sanjay Sundaresan"],
      venue:
        "International Journal of Advanced Research in Science, Communication and Technology (IJARSCT)",
      year: "2022",
      type: "Journal",
      pdf: "https://arxiv.org/abs/2210.07973",
      abstract:
        "Classification and identification of wild animals for tracking and protection purposes has become increasingly important with the deterioration of the environment, and technology is the agent of change which augments this process with novel solutions. Computer vision is one such technology which uses the abilities of artificial intelligence and machine learning models on visual inputs. Convolution neural networks (CNNs) have multiple layers which have different weights for the purpose of prediction of a particular input. The precedent for classification, however, is set by the image processing techniques which provide nearly ideal input images that produce optimal results. Image segmentation is one such widely used image processing method which provides a clear demarcation of the areas of interest in the image, be it regions or objects. The Efficiency of CNN can be related to the preprocessing done before training. Further, it is a well-established fact that heterogeneity in image sources is detrimental to the performance of CNNs. Thus, the added functionality of heterogeneity elimination is performed by the image processing techniques, introducing a level of consistency that sets the tone for the excellent feature extraction and eventually in classification.",
    },
  ],

  talks: [
    {
      title: "Behind the Compute: How LLMs Train, Infer, and Scale",
      event: "AMD Brown Bag Session",
      location: "San Jose, CA",
      date: "2025-11",
      type: "Talk",
      slides: "",
      video: "",
      description:
        "Presented a technical session on large language model workloads, exploring training strategies, parallelism approaches, and inference optimization, with insights from a MLPerf case study.",
    },
    {
      title: "SHARK-AI: AMD's Open-Source Inference Engine for LLMs",
      event: "AMD Brown Bag Session",
      location: "San Jose, CA",
      date: "2025-06",
      type: "Talk",
      slides: "",
      description:
        "Delivered a talk on end-to-end SHARK-AI workflows, covering PyTorch graph capture, model export, inference serving, IREE codegen, and GPU kernel compilation and execution.",
    },
    // {
    //   title: "The Future of AI Training Infrastructure",
    //   event: "ML Conf 2023",
    //   location: "Austin, TX",
    //   date: "2023-11",
    //   type: "Panel",
    //   video: "https://youtube.com/talk3",
    //   description:
    //     "Participated in panel discussion on emerging trends in AI training infrastructure.",
    // },
  ],

  blogPosts: [
    {
      title: "Optimizing LLM Inference: A Practical Guide",
      date: "2024-11",
      category: "AI Inference",
      keywords: ["LLM", "Optimization", "Performance", "Inference"],
      summary:
        "Deep dive into techniques for optimizing large language model inference, including quantization, batching, and caching strategies.",
      readTime: "12 min",
      link: "/blog/optimizing-llm-inference",
    },
    {
      title: "Distributed Training at Scale: Lessons Learned",
      date: "2024-09",
      category: "Training",
      keywords: [
        "Distributed Training",
        "Deep Learning",
        "Scalability",
        "Infrastructure",
      ],
      summary:
        "Sharing our experience scaling distributed training from single-node to multi-datacenter deployments.",
      readTime: "10 min",
      link: "/blog/distributed-training-scale",
    },
    {
      title: "Building a Modern MLOps Pipeline",
      date: "2024-07",
      category: "DevOps",
      keywords: ["MLOps", "CI/CD", "Automation", "Kubernetes"],
      summary:
        "Step-by-step guide to building an end-to-end MLOps pipeline with modern tools and best practices.",
      readTime: "15 min",
      link: "/blog/modern-mlops-pipeline",
    },
    {
      title: "System Design for AI Applications",
      date: "2024-05",
      category: "Software Engineering",
      keywords: ["System Design", "Architecture", "AI Systems", "Scalability"],
      summary:
        "Exploring architectural patterns and design considerations for building robust AI applications.",
      readTime: "11 min",
      link: "/blog/system-design-ai-apps",
    },
    {
      title: "Reducing AI Infrastructure Costs",
      date: "2024-03",
      category: "AI Inference",
      keywords: ["Cost Optimization", "Infrastructure", "Cloud", "Efficiency"],
      summary:
        "Practical strategies for reducing AI infrastructure costs while maintaining performance and reliability.",
      readTime: "8 min",
      link: "/blog/reducing-ai-costs",
    },
    {
      title: "The State of AI Training in 2024",
      date: "2024-01",
      category: "Training",
      keywords: ["AI Training", "Trends", "Hardware", "Techniques"],
      summary:
        "An overview of the current landscape of AI training, including hardware trends and emerging techniques.",
      readTime: "13 min",
      link: "/blog/state-of-ai-training-2024",
    },
  ],

  photography: [
    {
      location: "Swiss Alps, Switzerland",
      date: "2024-08",
      coverImage:
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
          title: "Sunrise at Matterhorn",
          description: "Golden hour illuminating the iconic Matterhorn peak",
        },
        {
          image: "https://images.unsplash.com/photo-1531366936337-7c912a4589a7",
          title: "Alpine Meadows",
          description: "Wildflowers blooming in the high alpine meadows",
        },
        {
          image: "https://images.unsplash.com/photo-1531973576160-7125cd663d86",
          title: "Mountain Lake",
          description:
            "Crystal clear glacial lake reflecting the surrounding peaks",
        },
        {
          image: "https://images.unsplash.com/photo-1434725039720-aaad6dd32dfe",
          title: "Cable Car Journey",
          description: "Ascending through the clouds to reach the summit",
        },
      ],
    },
    {
      location: "Tokyo, Japan",
      date: "2024-05",
      coverImage:
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf",
          title: "Shibuya Crossing",
          description: "The famous scramble crossing illuminated at night",
        },
        {
          image: "https://images.unsplash.com/photo-1513407030348-c983a97b98d8",
          title: "Neon Alleyways",
          description: "Narrow streets filled with colorful signs and lanterns",
        },
        {
          image: "https://images.unsplash.com/photo-1503899036084-c55cdd92da26",
          title: "Tokyo Tower at Dusk",
          description:
            "The iconic tower standing tall against the twilight sky",
        },
        {
          image: "https://images.unsplash.com/photo-1492571350019-22de08371fd3",
          title: "Traditional Temple",
          description: "Ancient architecture amidst the modern cityscape",
        },
        {
          image: "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc",
          title: "Cherry Blossom Season",
          description: "Sakura trees in full bloom along the river",
        },
      ],
    },
    {
      location: "Big Sur, California",
      date: "2024-03",
      coverImage:
        "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f",
          title: "Bixby Bridge",
          description: "The iconic bridge spanning the coastal canyon",
        },
        {
          image: "https://images.unsplash.com/photo-1501594907352-04cda38ebc29",
          title: "McWay Falls",
          description: "Waterfall cascading onto the pristine beach below",
        },
        {
          image: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
          title: "Coastal Sunset",
          description: "Sun setting over the Pacific Ocean horizon",
        },
      ],
    },
    {
      location: "Sahara Desert, Morocco",
      date: "2023-11",
      coverImage:
        "https://images.unsplash.com/photo-1509316785289-025f5b846b35",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1509316785289-025f5b846b35",
          title: "Dune Patterns",
          description: "Wind-sculpted sand creating mesmerizing patterns",
        },
        {
          image: "https://images.unsplash.com/photo-1473580044384-7ba9967e16a0",
          title: "Starry Night",
          description: "Milky Way visible above the desert dunes",
        },
        {
          image: "https://images.unsplash.com/photo-1511497584788-876760111969",
          title: "Camel Caravan",
          description: "Traditional transport crossing the vast desert",
        },
        {
          image: "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800",
          title: "Desert Sunrise",
          description: "First light painting the dunes in golden hues",
        },
      ],
    },
    {
      location: "Iceland",
      date: "2023-09",
      coverImage:
        "https://images.unsplash.com/photo-1504893524553-b855bce32c67",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1504893524553-b855bce32c67",
          title: "Northern Lights",
          description: "Aurora borealis dancing across the arctic sky",
        },
        {
          image: "https://images.unsplash.com/photo-1483347756197-71ef80e95f73",
          title: "Skógafoss Waterfall",
          description: "Massive waterfall cascading down the cliffs",
        },
        {
          image: "https://images.unsplash.com/photo-1476610182048-b716b8518aae",
          title: "Jökulsárlón Glacier",
          description: "Icebergs floating in the glacial lagoon",
        },
      ],
    },
    {
      location: "Santorini, Greece",
      date: "2023-07",
      coverImage:
        "https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff",
          title: "Oia Sunset",
          description: "White-washed buildings glowing in the sunset light",
        },
        {
          image: "https://images.unsplash.com/photo-1613395877344-13d4a8e0d49e",
          title: "Blue Domes",
          description: "Iconic blue-domed churches overlooking the caldera",
        },
        {
          image: "https://images.unsplash.com/photo-1533105079780-92b9be482077",
          title: "Aegean Views",
          description: "Dramatic cliffs meeting the deep blue sea",
        },
      ],
    },
    {
      location: "Patagonia, Argentina",
      date: "2023-04",
      coverImage:
        "https://images.unsplash.com/photo-1501594907352-04cda38ebc29",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1501594907352-04cda38ebc29",
          title: "Torres del Paine",
          description: "The famous granite towers piercing the sky",
        },
        {
          image: "https://images.unsplash.com/photo-1482192505345-5655af888803",
          title: "Perito Moreno Glacier",
          description: "Massive glacier walls in brilliant blue and white",
        },
        {
          image: "https://images.unsplash.com/photo-1519681393784-d120267933ba",
          title: "Mountain Reflection",
          description: "Peaks mirrored perfectly in the still lake",
        },
      ],
    },
    {
      location: "New Zealand",
      date: "2023-02",
      coverImage:
        "https://images.unsplash.com/photo-1469521669194-babb45599def",
      photos: [
        {
          image: "https://images.unsplash.com/photo-1469521669194-babb45599def",
          title: "Milford Sound",
          description: "Dramatic fjord surrounded by towering peaks",
        },
        {
          image: "https://images.unsplash.com/photo-1507699622108-4be3abd695ad",
          title: "Lake Tekapo",
          description: "Turquoise waters under the Southern Alps",
        },
        {
          image: "https://images.unsplash.com/photo-1589802829985-817e51171b92",
          title: "Glowworm Caves",
          description: "Magical bioluminescent ceiling in the caves",
        },
      ],
    },
  ],
};
