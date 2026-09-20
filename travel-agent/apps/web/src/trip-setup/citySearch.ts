import mainland from "../../../../config/china-cities.v1.json";
import curated from "../../../../config/cities.v1.json";

export interface DestinationCity {
  cityId: string;
  name: string;
  label: string;
  aliases: string[];
}

export const destinationCities: DestinationCity[] = mainland.cities.map(
  (city) => {
    const override = curated.cities.find(
      (item) => item.provider_codes.amap === city.adcode
    );
    return {
      cityId: override?.city_id ?? `cn-${city.adcode}`,
      name: city.official_name,
      label: `${city.province_name}-${city.official_name}`,
      aliases: [
        city.official_name,
        city.official_name.replace(/市$/, ""),
        city.province_name,
        ...(override?.aliases ?? [])
      ]
    };
  }
);

const normalize = (value: string) =>
  value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\s·._-]/g, "");

function oneEditApart(a: string, b: string): boolean {
  if (Math.abs(a.length - b.length) > 1) return false;
  let i = 0,
    j = 0,
    edits = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      i++;
      j++;
      continue;
    }
    if (++edits > 1) return false;
    if (a.length >= b.length) i++;
    if (b.length >= a.length) j++;
  }
  return edits + (a.length - i) + (b.length - j) <= 1;
}

export function searchDestinationCities(input: string): DestinationCity[] {
  const query = normalize(input);
  if (!query) return [];
  const ranked = destinationCities
    .map((city) => {
      const aliases = city.aliases.map(normalize);
      const score = aliases.some((name) => name === query)
        ? 0
        : aliases.some((name) => name.startsWith(query))
          ? 1
          : normalize(city.label).includes(query) ||
              aliases.some((name) => name.includes(query))
            ? 2
            : query.length >= 2 &&
                aliases.some((name) => oneEditApart(name, query))
              ? 3
              : 99;
      return { city, score };
    })
    .filter(({ score }) => score < 99)
    .sort(
      (a, b) => a.score - b.score || a.city.cityId.localeCompare(b.city.cityId)
    );
  const relevant = ranked.some(({ score }) => score < 3)
    ? ranked.filter(({ score }) => score < 3)
    : ranked;
  return relevant.slice(0, 6).map(({ city }) => city);
}
