import React from "react";

interface SparklineProps {
  data?: number[] | null;
  width?: number;
  height?: number;
  className?: string;
}

const Sparkline: React.FC<SparklineProps> = ({ data, width = 120, height = 32, className }) => {
  if (!data || data.length === 0) {
    return <span className="text-muted-foreground text-xs">Not present in document</span>;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1 || 1)) * width;
    const y = height - ((d - min) / range) * height;
    return `${x},${y}`;
  });

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={className} aria-label="sparkline">
      <polyline
        fill="none"
        stroke="currentColor"
        strokeOpacity={0.85}
        strokeWidth={2}
        points={points.join(" ")}
      />
    </svg>
  );
};

export default Sparkline;