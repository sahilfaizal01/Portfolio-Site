export const portfolioData = {
  profile: {
    name: "Your Name",
    title: "Tech Lead & AI TPM",
    tagline: "Building scalable AI systems and leading technical teams",
    email: "your.email@example.com",
    linkedin: "https://linkedin.com/in/yourprofile",
    github: "https://github.com/yourprofile",
    twitter: "https://twitter.com/yourprofile"
  },

  skills: [
    { category: "AI/ML", items: ["Machine Learning", "Deep Learning", "LLM Optimization", "AI Infrastructure"] },
    { category: "Leadership", items: ["Technical Program Management", "Team Leadership", "Strategic Planning", "Stakeholder Management"] },
    { category: "Engineering", items: ["Python", "Distributed Systems", "Cloud Architecture", "DevOps"] },
    { category: "Specializations", items: ["AI Inference", "Model Training", "MLOps", "Performance Optimization"] }
  ],

  achievements: [
    "Led AI infrastructure migration serving 10M+ daily requests",
    "Reduced model inference latency by 60% through optimization",
    "Built and scaled engineering team from 5 to 25 members",
    "Published 15+ papers in top-tier AI conferences"
  ],

  experiences: [
    {
      company: "Tech Company Inc.",
      position: "Senior Tech Lead - AI Platform",
      location: "San Francisco, CA",
      startDate: "2022-01",
      endDate: "Present",
      description: "Leading the AI infrastructure team, responsible for building and scaling ML serving infrastructure. Managing cross-functional teams and driving technical roadmap for AI platform services.",
      highlights: [
        "Architected distributed inference system handling 50M+ requests/day",
        "Led team of 12 engineers across ML infrastructure and platform",
        "Reduced infrastructure costs by 40% through optimization initiatives"
      ]
    },
    {
      company: "AI Startup Labs",
      position: "AI Technical Program Manager",
      location: "Remote",
      startDate: "2020-03",
      endDate: "2021-12",
      description: "Managed end-to-end delivery of AI/ML products, coordinating between research, engineering, and product teams.",
      highlights: [
        "Delivered 5 major AI product releases on time and within budget",
        "Established MLOps practices reducing deployment time by 70%",
        "Drove adoption of model monitoring and observability tools"
      ]
    },
    {
      company: "Enterprise Solutions Corp",
      position: "Senior Software Engineer",
      location: "New York, NY",
      startDate: "2018-06",
      endDate: "2020-02",
      description: "Developed scalable backend systems and APIs for enterprise applications.",
      highlights: [
        "Built microservices architecture serving 1M+ users",
        "Implemented CI/CD pipelines improving deployment frequency by 10x",
        "Mentored junior engineers and led code review practices"
      ]
    }
  ],

  projects: [
    {
      title: "LLM Inference Optimizer",
      description: "Open-source toolkit for optimizing large language model inference performance",
      technologies: ["Python", "PyTorch", "CUDA", "Docker"],
      github: "https://github.com/yourprofile/llm-optimizer",
      demo: "https://demo.example.com",
      highlights: [
        "Achieved 3x speedup in inference latency",
        "Support for multiple model architectures",
        "1000+ GitHub stars"
      ]
    },
    {
      title: "AI Training Pipeline",
      description: "Distributed training framework for computer vision models at scale",
      technologies: ["Python", "TensorFlow", "Kubernetes", "Ray"],
      github: "https://github.com/yourprofile/training-pipeline",
      highlights: [
        "Scaled training to 100+ GPUs",
        "Automated hyperparameter tuning",
        "Reduced training time by 50%"
      ]
    },
    {
      title: "MLOps Dashboard",
      description: "Real-time monitoring and observability platform for ML models in production",
      technologies: ["React", "Python", "PostgreSQL", "Grafana"],
      demo: "https://mlops-demo.example.com",
      highlights: [
        "Tracks 100+ models in production",
        "Real-time performance metrics",
        "Automated alerting system"
      ]
    }
  ],

  publications: [
    {
      title: "Efficient Inference Strategies for Large Language Models",
      authors: ["Your Name", "Co-Author 1", "Co-Author 2"],
      venue: "NeurIPS 2024",
      year: "2024",
      type: "Conference",
      location: "New Orleans, LA",
      pdf: "https://arxiv.org/paper1",
      abstract: "We present novel techniques for optimizing inference performance of large language models..."
    },
    {
      title: "Scaling Distributed Training with Dynamic Resource Allocation",
      authors: ["Your Name", "Co-Author 1"],
      venue: "ICML 2023",
      year: "2023",
      type: "Conference",
      location: "Honolulu, HI",
      pdf: "https://arxiv.org/paper2",
      abstract: "This work introduces a dynamic resource allocation strategy for distributed training..."
    },
    {
      title: "Production ML Systems: Lessons from the Trenches",
      authors: ["Your Name"],
      venue: "ACM Queue",
      year: "2023",
      type: "Journal",
      pdf: "https://queue.acm.org/paper3",
      abstract: "A comprehensive overview of best practices for deploying ML systems in production..."
    }
  ],

  talks: [
    {
      title: "Building Reliable AI Infrastructure at Scale",
      event: "AI Engineering Summit 2024",
      location: "San Francisco, CA",
      date: "2024-10",
      type: "Keynote",
      slides: "https://slides.example.com/talk1",
      video: "https://youtube.com/talk1",
      description: "Discussed strategies for building and maintaining AI infrastructure that serves millions of users."
    },
    {
      title: "MLOps Best Practices: From Research to Production",
      event: "DevOps Days 2024",
      location: "Seattle, WA",
      date: "2024-06",
      type: "Talk",
      slides: "https://slides.example.com/talk2",
      description: "Shared practical insights on bridging the gap between ML research and production deployment."
    },
    {
      title: "The Future of AI Training Infrastructure",
      event: "ML Conf 2023",
      location: "Austin, TX",
      date: "2023-11",
      type: "Panel",
      video: "https://youtube.com/talk3",
      description: "Participated in panel discussion on emerging trends in AI training infrastructure."
    }
  ],

  blogPosts: [
    {
      title: "Optimizing LLM Inference: A Practical Guide",
      date: "2024-11",
      category: "AI Inference",
      keywords: ["LLM", "Optimization", "Performance", "Inference"],
      summary: "Deep dive into techniques for optimizing large language model inference, including quantization, batching, and caching strategies.",
      readTime: "12 min",
      link: "/blog/optimizing-llm-inference"
    },
    {
      title: "Distributed Training at Scale: Lessons Learned",
      date: "2024-09",
      category: "Training",
      keywords: ["Distributed Training", "Deep Learning", "Scalability", "Infrastructure"],
      summary: "Sharing our experience scaling distributed training from single-node to multi-datacenter deployments.",
      readTime: "10 min",
      link: "/blog/distributed-training-scale"
    },
    {
      title: "Building a Modern MLOps Pipeline",
      date: "2024-07",
      category: "DevOps",
      keywords: ["MLOps", "CI/CD", "Automation", "Kubernetes"],
      summary: "Step-by-step guide to building an end-to-end MLOps pipeline with modern tools and best practices.",
      readTime: "15 min",
      link: "/blog/modern-mlops-pipeline"
    },
    {
      title: "System Design for AI Applications",
      date: "2024-05",
      category: "Software Engineering",
      keywords: ["System Design", "Architecture", "AI Systems", "Scalability"],
      summary: "Exploring architectural patterns and design considerations for building robust AI applications.",
      readTime: "11 min",
      link: "/blog/system-design-ai-apps"
    },
    {
      title: "Reducing AI Infrastructure Costs",
      date: "2024-03",
      category: "AI Inference",
      keywords: ["Cost Optimization", "Infrastructure", "Cloud", "Efficiency"],
      summary: "Practical strategies for reducing AI infrastructure costs while maintaining performance and reliability.",
      readTime: "8 min",
      link: "/blog/reducing-ai-costs"
    },
    {
      title: "The State of AI Training in 2024",
      date: "2024-01",
      category: "Training",
      keywords: ["AI Training", "Trends", "Hardware", "Techniques"],
      summary: "An overview of the current landscape of AI training, including hardware trends and emerging techniques.",
      readTime: "13 min",
      link: "/blog/state-of-ai-training-2024"
    }
  ],

  photography: [
    {
      title: "Mountain Peaks",
      location: "Swiss Alps, Switzerland",
      date: "2024-08",
      image: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
      description: "Sunrise over the majestic peaks of the Swiss Alps"
    },
    {
      title: "Urban Exploration",
      location: "Tokyo, Japan",
      date: "2024-05",
      image: "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf",
      description: "Neon-lit streets of Shibuya at night"
    },
    {
      title: "Coastal Wanderings",
      location: "Big Sur, California",
      date: "2024-03",
      image: "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f",
      description: "Dramatic coastline meeting the Pacific Ocean"
    },
    {
      title: "Desert Solitude",
      location: "Sahara, Morocco",
      date: "2023-11",
      image: "https://images.unsplash.com/photo-1509316785289-025f5b846b35",
      description: "Endless dunes under a starlit sky"
    }
  ]
};
