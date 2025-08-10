import React from "react";
import { cn } from "@/lib/utils";

interface FieldRowProps {
  label: string;
  value?: React.ReactNode | string | number | null;
  fallback?: React.ReactNode;
  className?: string;
}

const FieldRow: React.FC<FieldRowProps> = ({ label, value, fallback = "Not present in document", className }) => {
  const display = value === null || value === undefined || value === "" ? (
    <span className="text-muted-foreground">{fallback}</span>
  ) : (
    value
  );

  return (
    <div className={cn("flex items-baseline justify-between gap-4 rounded-xl bg-card/50 px-3 py-2 ring-1 ring-inset ring-border", className)}>
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="font-medium">{display}</span>
    </div>
  );
};

export default FieldRow;
