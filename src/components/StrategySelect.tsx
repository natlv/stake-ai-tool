import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const StrategySelect = () => {
  return (
    <div className="w-full max-w-sm">
      <label htmlFor="strategy" className="mb-2 block text-sm text-muted-foreground">
        Select your investment strategy
      </label>
      <Select>
        <SelectTrigger id="strategy" className="w-full">
          <SelectValue placeholder="Choose a strategy" />
        </SelectTrigger>
        <SelectContent className="z-50 bg-popover text-popover-foreground">
          <SelectGroup>
            
            <SelectItem value="conservative">Conservative</SelectItem>
            <SelectItem value="balanced">Balanced</SelectItem>
            <SelectItem value="aggressive">Aggressive</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
};

export default StrategySelect;
