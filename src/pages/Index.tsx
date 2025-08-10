import React from "react";
import StrategySelect from "@/components/StrategySelect";
import FileUploadCard from "@/components/FileUploadCard";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { analyzePdf } from "@/services/analyze";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [isValid, setIsValid] = React.useState(false);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);
  const navigate = useNavigate();

  const handleFileChange = (
    file: File | null,
    meta: { isValid: boolean; name?: string; size?: number }
  ) => {
    setSelectedFile(file);
    setIsValid(meta.isValid);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || !isValid) return;
    setIsAnalyzing(true);
    try {
      const result = await analyzePdf(selectedFile);
      toast({ title: "Analysis complete", description: "Now showing results." });
      navigate("/results", { state: { result } });
    } catch (e) {
      toast({ title: "Analysis failed", description: "Please try again.", variant: "destructive" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main>
      <section className="container mx-auto min-h-screen px-6 py-16">
          <header className="mb-10 text-center md:text-left">
            <h1 className="mb-3 whitespace-nowrap text-3xl font-bold tracking-tight md:text-4xl">
              Upload Financial Docs for Investment Strategy
            </h1>
            <p className="text-muted-foreground">Choose a strategy and upload your PDFs. We only accept .pdf files.</p>
          </header>

          <div className="mb-6 max-w-sm">
            <StrategySelect />
          </div>

          <div className="max-w-3xl">
            <FileUploadCard isAnalyzing={isAnalyzing} onFileChange={handleFileChange} />
          </div>

          <div className="mt-6 flex justify-start">
            <Button
              size="lg"
              variant="hero"
              onClick={handleAnalyze}
              disabled={!isValid || !selectedFile || isAnalyzing}
              aria-disabled={!isValid || !selectedFile || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="animate-spin" /> Analyzing...
                </>
              ) : (
                <>Analyze</>
              )}
            </Button>
          </div>
      </section>
    </main>
  );
};

export default Index;
