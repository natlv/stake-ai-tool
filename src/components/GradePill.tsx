import React from "react";

export type Grade = "A" | "A-" | "B+" | "B" | "B-" | "C+" | "C" | "C-" | "D";

interface GradePillProps {
  grade: Grade;
  className?: string;
}

const colorFor = (g: Grade) => {
  if (g.startsWith("A")) return "brand-green" as const;
  if (g.startsWith("B")) return "brand-teal" as const;
  if (g.startsWith("C")) return "brand-cyan" as const;
  return "destructive" as const;
};

const GradePill: React.FC<GradePillProps> = ({ grade, className }) => {
  const c = colorFor(grade);
  const base =
    "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset transition-colors";
  const styles =
    c === "destructive"
      ? "bg-[hsl(var(--destructive)_/_0.14)] text-[hsl(var(--destructive))] ring-[hsl(var(--destructive)_/_0.35)] hover:bg-[hsl(var(--destructive)_/_0.22)]"
      : `bg-[hsl(var(--${c})_/_0.14)] text-[hsl(var(--${c}))] ring-[hsl(var(--${c})_/_0.35)] hover:bg-[hsl(var(--${c})_/_0.22)]`;

  return (
    <span className={`${base} ${styles} ${className || ""}`} aria-label={`Confidence grade: ${grade}`}>
      {grade}
    </span>
  );
};

export default GradePill;
