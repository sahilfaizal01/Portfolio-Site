import React from 'react';
import Experience from '../components/Experience';
import { portfolioData } from '../data/portfolioData';

const ExperiencePage = () => {
  return (
    <div className="pt-20 min-h-screen">
      <Experience experiences={portfolioData.experiences} />
    </div>
  );
};

export default ExperiencePage;
