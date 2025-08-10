import StrategySelect from "@/components/StrategySelect";
import FileUploadCard from "@/components/FileUploadCard";

const Index = () => {
  return (
    <main>
      <section className="container mx-auto min-h-screen px-6 py-16">
        <header className="mx-auto mb-12 max-w-3xl text-center">
          <h1 className="mb-3 text-4xl font-bold tracking-tight">Upload Financial Documents for Investment Strategy</h1>
          <p className="text-muted-foreground">Choose a strategy and upload your PDFs. We only accept .pdf files.</p>
        </header>

        <div className="mx-auto mb-10 max-w-3xl">
          <StrategySelect />
        </div>

        <div className="mx-auto max-w-3xl">
          <FileUploadCard />
        </div>
      </section>
    </main>
  );
};

export default Index;
