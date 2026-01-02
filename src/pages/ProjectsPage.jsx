import React from 'react';
import Projects from '../components/Projects';
import { portfolioData } from '../data/portfolioData';

const ProjectsPage = () => {
  return (
    <div className="pt-20 min-h-screen">
      <Projects projects={portfolioData.projects} />
    </div>
  );
};

export default ProjectsPage;
