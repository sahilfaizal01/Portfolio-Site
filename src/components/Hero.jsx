import React from 'react';
import { ArrowDown, Award } from 'lucide-react';

const Hero = ({ profile, skills, achievements }) => {
  return (
    <section id="about" className="min-h-screen flex items-center section-padding pt-32">
      <div className="container-width w-full">
        <div className="max-w-4xl">
          <h1 className="text-5xl md:text-7xl font-bold text-dark-text-primary mb-4">
            {profile.name}
          </h1>
          <h2 className="text-2xl md:text-3xl text-dark-text-secondary mb-6">
            {profile.title}
          </h2>
          <p className="text-lg md:text-xl text-dark-text-muted mb-12 max-w-2xl">
            {profile.tagline}
          </p>

          {/* Skills */}
          <div className="mb-12">
            <h3 className="text-xl font-semibold text-dark-text-primary mb-6 flex items-center">
              <span className="w-12 h-px bg-dark-border mr-4"></span>
              Core Skills
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {skills.map((skillGroup, index) => (
                <div
                  key={index}
                  className="bg-dark-surface border border-dark-border rounded-lg p-6 hover:border-dark-text-muted transition-colors"
                >
                  <h4 className="text-lg font-semibold text-dark-text-primary mb-3">
                    {skillGroup.category}
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {skillGroup.items.map((skill, idx) => (
                      <span
                        key={idx}
                        className="text-sm text-dark-text-secondary bg-dark-bg px-3 py-1 rounded-full border border-dark-border"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Achievements */}
          <div>
            <h3 className="text-xl font-semibold text-dark-text-primary mb-6 flex items-center">
              <Award className="mr-3" size={24} />
              Key Achievements
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {achievements.map((achievement, index) => (
                <div
                  key={index}
                  className="flex items-start space-x-3 bg-dark-surface border border-dark-border rounded-lg p-4"
                >
                  <div className="w-2 h-2 bg-dark-text-secondary rounded-full mt-2 flex-shrink-0"></div>
                  <p className="text-dark-text-secondary">{achievement}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Scroll Indicator */}
          <div className="mt-16 flex justify-center">
            <a
              href="#experience"
              onClick={(e) => {
                e.preventDefault();
                document.querySelector('#experience').scrollIntoView({
                  behavior: 'smooth',
                  block: 'start'
                });
              }}
              className="text-dark-text-muted hover:text-dark-text-secondary transition-colors animate-bounce"
            >
              <ArrowDown size={32} />
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
