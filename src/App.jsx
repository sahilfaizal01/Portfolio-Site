import React from 'react';
import Navigation from './components/Navigation';
import Hero from './components/Hero';
import Experience from './components/Experience';
import Projects from './components/Projects';
import Publications from './components/Publications';
import Talks from './components/Talks';
import Blog from './components/Blog';
import Photography from './components/Photography';
import Contact from './components/Contact';
import { portfolioData } from './data/portfolioData';

function App() {
  return (
    <div className="min-h-screen bg-dark-bg">
      <Navigation />

      <main>
        <Hero
          profile={portfolioData.profile}
          skills={portfolioData.skills}
          achievements={portfolioData.achievements}
        />

        <Experience experiences={portfolioData.experiences} />

        <Projects projects={portfolioData.projects} />

        <Publications publications={portfolioData.publications} />

        <Talks talks={portfolioData.talks} />

        <Blog blogPosts={portfolioData.blogPosts} />

        <Photography photography={portfolioData.photography} />

        <Contact profile={portfolioData.profile} />
      </main>
    </div>
  );
}

export default App;
