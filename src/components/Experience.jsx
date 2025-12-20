import React from 'react';
import { Briefcase, MapPin, Calendar } from 'lucide-react';

const Experience = ({ experiences }) => {
  const formatDate = (dateString) => {
    if (dateString === 'Present') return 'Present';
    const [year, month] = dateString.split('-');
    const date = new Date(year, month - 1);
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };

  const calculateDuration = (start, end) => {
    const startDate = new Date(start);
    const endDate = end === 'Present' ? new Date() : new Date(end);
    const months = (endDate.getFullYear() - startDate.getFullYear()) * 12 +
                   (endDate.getMonth() - startDate.getMonth());
    const years = Math.floor(months / 12);
    const remainingMonths = months % 12;

    if (years === 0) return `${remainingMonths} mo`;
    if (remainingMonths === 0) return `${years} yr`;
    return `${years} yr ${remainingMonths} mo`;
  };

  return (
    <section id="experience" className="section-padding bg-dark-surface">
      <div className="container-width">
        <h2 className="text-4xl md:text-5xl font-bold text-dark-text-primary mb-4">
          Experience
        </h2>
        <div className="w-20 h-1 bg-dark-border mb-12"></div>

        <div className="space-y-12">
          {experiences.map((exp, index) => (
            <div
              key={index}
              className="relative pl-8 pb-12 border-l-2 border-dark-border last:pb-0"
            >
              {/* Timeline Dot */}
              <div className="absolute left-0 top-0 w-4 h-4 bg-dark-text-primary rounded-full -translate-x-[9px] border-4 border-dark-bg"></div>

              {/* Content */}
              <div className="bg-dark-bg border border-dark-border rounded-lg p-6 hover:border-dark-text-muted transition-colors">
                <div className="flex flex-col md:flex-row md:items-start md:justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold text-dark-text-primary mb-2">
                      {exp.position}
                    </h3>
                    <div className="flex items-center text-dark-text-secondary mb-2">
                      <Briefcase size={16} className="mr-2" />
                      <span className="font-semibold">{exp.company}</span>
                    </div>
                  </div>

                  <div className="mt-3 md:mt-0 md:text-right">
                    <div className="flex items-center text-dark-text-muted mb-1 md:justify-end">
                      <Calendar size={16} className="mr-2" />
                      <span className="text-sm">
                        {formatDate(exp.startDate)} - {formatDate(exp.endDate)}
                      </span>
                    </div>
                    <span className="text-sm text-dark-text-muted">
                      {calculateDuration(exp.startDate, exp.endDate)}
                    </span>
                  </div>
                </div>

                <div className="flex items-center text-dark-text-muted mb-4">
                  <MapPin size={16} className="mr-2" />
                  <span className="text-sm">{exp.location}</span>
                </div>

                <p className="text-dark-text-secondary mb-4 leading-relaxed">
                  {exp.description}
                </p>

                <div className="space-y-2">
                  {exp.highlights.map((highlight, idx) => (
                    <div key={idx} className="flex items-start">
                      <span className="text-dark-text-muted mr-3 mt-1.5">▸</span>
                      <p className="text-dark-text-secondary">{highlight}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Experience;
