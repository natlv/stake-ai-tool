import { Link } from "react-router-dom";

const SiteHeader = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <nav className="container mx-auto flex h-14 items-center justify-between px-6 md:h-16">
        <Link to="/" className="inline-flex items-center gap-3 select-none" aria-label="Stake.ai home">
          <img
            src="/lovable-uploads/f3ddb896-1a9e-4533-a9fa-a39562c35d78.png"
            alt="Stake.ai logo"
            className="h-7 w-7 object-contain"
            loading="eager"
          />
          <span className="text-base font-semibold tracking-tight md:text-lg">stake.ai</span>
        </Link>
      </nav>
    </header>
  );
};

export default SiteHeader;
