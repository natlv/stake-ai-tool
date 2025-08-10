import React from "react";

export type DecisionValue = "BUY" | "SELL";

interface BadgeDecisionProps {
  value: DecisionValue;
  className?: string;
}

const BadgeDecision: React.FC<BadgeDecisionProps> = ({ value, className }) => {
  const isBuy = value === "BUY";
  const base =
    "inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ring-inset transition-colors";
  const buyClasses =
    "bg-[hsl(var(--brand-green)_/_0.14)] text-[hsl(var(--brand-green))] ring-[hsl(var(--brand-green)_/_0.35)] hover:bg-[hsl(var(--brand-green)_/_0.22)]";
  const sellClasses =
    "bg-[hsl(var(--destructive)_/_0.14)] text-[hsl(var(--destructive))] ring-[hsl(var(--destructive)_/_0.35)] hover:bg-[hsl(var(--destructive)_/_0.22)]";

  return (
    <span
      className={`${base} ${isBuy ? buyClasses : sellClasses} ${className || ""}`}
      aria-label={`Decision: ${value}`}
    >
      {value}
    </span>
  );
};

export default BadgeDecision;
