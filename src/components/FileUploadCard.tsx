import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FileText, UploadCloud, Loader2 } from "lucide-react";
import { toast } from "@/hooks/use-toast";

const FileUploadCard: React.FC = () => {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = React.useState(false);
  const [fileName, setFileName] = React.useState<string | null>(null);
  const [fileSize, setFileSize] = React.useState<number | null>(null);
  const [isValid, setIsValid] = React.useState(false);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);

  const onSelectFile = (file: File) => {
    const name = file?.name || "";
    const mimeOk = file?.type === "application/pdf";
    const extOk = name.toLowerCase().endsWith(".pdf");
    const isPdf = Boolean(file && mimeOk && extOk);
    if (!isPdf) {
      setFileName(null);
      setFileSize(null);
      setIsValid(false);
      setIsAnalyzing(false);
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF document (.pdf)",
        variant: "destructive",
      });
      return;
    }
    setFileName(name);
    setFileSize(file.size);
    setIsValid(true);
    setIsAnalyzing(false);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      onSelectFile(files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onSelectFile(file);
  };

  return (
    <Card className="mx-auto w-full max-w-2xl border border-border/60 bg-card/60 backdrop-blur-sm">
      <CardContent className="p-8">
        <div
          role="button"
          tabIndex={0}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
            isDragging ? "border-brand-cyan/70 bg-accent/20" : "border-muted/40 hover:border-brand-teal/60"
          }`}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
          aria-label="Upload PDF by drag-and-drop or click to choose"
        >
          <div className="mb-6 flex items-center justify-center rounded-full bg-background/60 p-5 shadow-sm">
            <UploadCloud className="text-brand-cyan" size={42} />
          </div>
          <p className="mb-2 text-lg font-medium">Upload Financial Documents</p>
          <p className="mb-6 text-sm text-muted-foreground">Drag & drop your PDF here, or click to browse</p>
          <div className="flex items-center gap-3">
            <Button variant="hero" size="lg">
              <FileText /> Choose PDF
            </Button>
            <Button
              size="lg"
              variant="secondary"
              onClick={(e) => {
                e.stopPropagation();
                if (!isValid) return;
                setIsAnalyzing(true);
              }}
              disabled={!isValid || isAnalyzing}
              aria-disabled={!isValid || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="animate-spin" /> Analyzing...
                </>
              ) : (
                <>Analyze</>
              )}
            </Button>
            <input
              ref={inputRef}
              type="file"
              accept=".pdf,application/pdf"
              className="hidden"
              onChange={onInputChange}
            />
          </div>
          {fileName && (
            <p className="mt-4 text-sm text-muted-foreground">
              Selected: {fileName}
              {fileSize !== null ? ` (${(fileSize / 1024 / 1024).toFixed(2)} MB)` : ""}
            </p>
          )}
          {isAnalyzing && (
            <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
              <Loader2 className="animate-spin text-brand-cyan" />
              <span>Analyzing document...</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default FileUploadCard;
