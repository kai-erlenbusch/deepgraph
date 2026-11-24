import type {
    ColorScaleName,
    DeepgraphPreset,
    DeepgraphPresetResolver,
    ExperimentalOptions,
    TileSource,
    ViewOptionsPatch,
} from '../index.ts';

export const GAIA_FALLBACK_STATS: Record<string, { min: number; max: number }> = {
    phot_g_mean_mag: { min: 0.0, max: 21.0 },
    abs_m: { min: -2.0, max: 18.0 },
    bp_rp: { min: -0.5, max: 4.0 },
    radial_velocity: { min: -150.0, max: 150.0 },
    teff_gspphot: { min: 3000.0, max: 10000.0 },
    parallax: { min: 0.0, max: 20.0 },
    pmra: { min: -100.0, max: 100.0 },
    pmdec: { min: -100.0, max: 100.0 },
};

export const GAIA_DEMO_PRESET: DeepgraphPreset = {
    name: 'gaia-dr3',
    fallbackRootBounds: { minX: -2.0, maxX: 2.0, minY: -1.0, maxY: 1.0 },
    fallbackStats: GAIA_FALLBACK_STATS,
    renderPipeline: 'optical',
    defaultEncoding: {
        color: { field: 'bp_rp', scale: 'viridis', domain: [-0.5, 4.0] },
        size: { field: 'phot_g_mean_mag', domain: [-2.0, 18.0] },
    },
    experimental: {
        magnitudeFields: ['phot_g_mean_mag', 'abs_m'],
        opticalColorField: 'bp_rp',
        legacyFlipY: false,
    },
};

export function isGaiaSourceHint(source: TileSource): boolean {
    let rawName = '';
    if (typeof source === 'string') {
        rawName = source;
    } else if (typeof File !== 'undefined' && source instanceof File) {
        rawName = source.name;
    } else if ('name' in source && typeof (source as { name?: unknown }).name === 'string') {
        rawName = String((source as { name?: string }).name);
    } else {
        const custom = source as { key?: string; getKey?: () => string };
        rawName = custom.key ?? custom.getKey?.() ?? '';
    }
    return rawName.toLowerCase().includes('gaia');
}

export function isGaiaMetadata(rawMetadata: Record<string, any> | undefined): boolean {
    if (!rawMetadata) return false;
    const viewObj =
        typeof rawMetadata.view === 'object' && rawMetadata.view !== null
            ? rawMetadata.view
            : rawMetadata;
    return viewObj?.mode === 'Gaia Baseline';
}

/** Preset resolver invoked inside `DatasetSession.open()` right after reading archive metadata. */
export const resolveDemoPreset: DeepgraphPresetResolver = (rawMetadata, source) => {
    if (isGaiaSourceHint(source) || isGaiaMetadata(rawMetadata as Record<string, any>)) {
        return GAIA_DEMO_PRESET;
    }
    return undefined;
};

export const GAIA_SCIENCE_LENS_NAMES = [
    '10. Sky in Colour (Optical Flux)',
    '1. Stellar Motion (Star Trails +400 kyr)',
    '2. Stellar Age (FLAME / Teff Map)',
    '3. Brightness & Dust (G - GRVS Reddening)',
    '4. Variable Stars (RR Lyrae Metallicity)',
    '5. Stellar Extinction (GSP-Phot A0)',
    '6. Interstellar Medium (DIB HEALPix-5)',
    '7. Radial Velocities (Doppler Map)',
    '8. Chemical Map (Composition [M/H])',
    '9. 3D Stellar Motion (Vr + PM Streamlines)',
    'Hertzsprung-Russell (HR) Diagram',
    'Custom Chart Mode',
] as const;

export interface ScienceLensApplication {
    modeLabel: 'Gaia Baseline' | 'Chart Mode';
    aggregationLabel?: 'Mean Field (HEALPix-style)' | 'Additive Points';
    patch: ViewOptionsPatch;
    sliderColorBounds?: [number, number];
}

export function buildScienceLensPatch(
    lens: string,
    gaiaMagCol: string,
    trailStyle: string,
    currentAggregationLabel: string
): ScienceLensApplication {
    const resetFlipY =
        lens !== 'Hertzsprung-Russell (HR) Diagram' && lens !== 'Custom Chart Mode'
            ? { flipY: false }
            : {};

    if (lens === '10. Sky in Colour (Optical Flux)') {
        return {
            modeLabel: 'Gaia Baseline',
            patch: {
                ...resetFlipY,
                pipeline: 'optical',
                xField: null,
                yField: null,
                colorField: 'bp_rp',
                sizeField: gaiaMagCol,
                lightBackground: false,
                invertScale: false,
                experimental: {
                    metricType: 0,
                    auxFields: ['', '', ''],
                },
            },
        };
    }

    if (lens === '1. Stellar Motion (Star Trails +400 kyr)') {
        return {
            modeLabel: 'Chart Mode',
            patch: {
                ...resetFlipY,
                pipeline: 'cartesian',
                xField: null,
                yField: null,
                colorField: 'bp_rp',
                sizeField: gaiaMagCol,
                lightBackground: false,
                invertScale: false,
                experimental: {
                    aggregation: 'trails',
                    metricType: 0,
                    auxFields: ['pmra', 'pmdec', 'parallax'],
                    trailTimeKyr:
                        trailStyle === '1.6 Myr Time Evolution (80 kyr Comets)' ? 80 : 400,
                    minParallax: 8.0,
                },
            },
        };
    }

    const makeMeanField = (cfg: {
        metricType: number;
        auxFields: [string, string, string];
        colorField: string;
        colorScale: ColorScaleName | NonNullable<ExperimentalOptions['colorScale']>;
        colorMin: number;
        colorMax: number;
        sliderBounds: [number, number];
        smoothingKernel: number;
        invertScale: boolean;
        lightBackground: boolean;
    }): ScienceLensApplication => ({
        modeLabel: 'Chart Mode',
        aggregationLabel: 'Mean Field (HEALPix-style)',
        sliderColorBounds: cfg.sliderBounds,
        patch: {
            ...resetFlipY,
            pipeline: 'cartesian',
            aggregation: 'mean_field',
            xField: null,
            yField: null,
            colorField: cfg.colorField,
            colorScale: cfg.colorScale,
            colorMin: cfg.colorMin,
            colorMax: cfg.colorMax,
            smoothingKernel: cfg.smoothingKernel,
            invertScale: cfg.invertScale,
            lightBackground: cfg.lightBackground,
            experimental: {
                metricType: cfg.metricType,
                auxFields: cfg.auxFields,
            },
        },
    });

    if (lens === '2. Stellar Age (FLAME / Teff Map)') {
        return makeMeanField({
            metricType: 6,
            auxFields: ['', '', ''],
            colorField: 'teff_gspphot',
            colorScale: 'esa_age',
            colorMin: 5400.0,
            colorMax: 7100.0,
            sliderBounds: [3500.0, 10000.0],
            smoothingKernel: 1,
            invertScale: true,
            lightBackground: true,
        });
    }
    if (lens === '3. Brightness & Dust (G - GRVS Reddening)') {
        return makeMeanField({
            metricType: 1,
            auxFields: ['teff_gspphot', '', ''],
            colorField: 'bp_rp',
            colorScale: 'esa_dust',
            colorMin: 0.08,
            colorMax: 1.95,
            sliderBounds: [0.0, 4.0],
            smoothingKernel: 1,
            invertScale: false,
            lightBackground: true,
        });
    }
    if (lens === '4. Variable Stars (RR Lyrae Metallicity)') {
        return makeMeanField({
            metricType: 4,
            auxFields: ['teff_gspphot', '', 'parallax'],
            colorField: 'bp_rp',
            colorScale: 'jet',
            colorMin: -1.95,
            colorMax: 0.05,
            sliderBounds: [-2.5, 0.5],
            smoothingKernel: 0,
            invertScale: false,
            lightBackground: true,
        });
    }
    if (lens === '5. Stellar Extinction (GSP-Phot A0)') {
        return makeMeanField({
            metricType: 1,
            auxFields: ['teff_gspphot', '', ''],
            colorField: 'bp_rp',
            colorScale: 'esa_extinction',
            colorMin: 0.12,
            colorMax: 2.15,
            sliderBounds: [0.0, 4.5],
            smoothingKernel: 1,
            invertScale: false,
            lightBackground: false,
        });
    }
    if (lens === '6. Interstellar Medium (DIB HEALPix-5)') {
        return makeMeanField({
            metricType: 3,
            auxFields: ['teff_gspphot', '', ''],
            colorField: 'bp_rp',
            colorScale: 'esa_dust',
            colorMin: 0.02,
            colorMax: 0.28,
            sliderBounds: [0.0, 0.8],
            smoothingKernel: 2,
            invertScale: false,
            lightBackground: true,
        });
    }
    if (lens === '7. Radial Velocities (Doppler Map)') {
        return makeMeanField({
            metricType: 5,
            auxFields: ['', '', 'parallax'],
            colorField: 'radial_velocity',
            colorScale: 'magma',
            colorMin: -42.0,
            colorMax: 42.0,
            sliderBounds: [-120.0, 120.0],
            smoothingKernel: 1,
            invertScale: false,
            lightBackground: false,
        });
    }
    if (lens === '8. Chemical Map (Composition [M/H])') {
        return makeMeanField({
            metricType: 2,
            auxFields: ['teff_gspphot', '', 'radial_velocity'],
            colorField: 'bp_rp',
            colorScale: 'jet',
            colorMin: -0.68,
            colorMax: 0.28,
            sliderBounds: [-1.8, 0.8],
            smoothingKernel: 1,
            invertScale: false,
            lightBackground: true,
        });
    }
    if (lens === '9. 3D Stellar Motion (Vr + PM Streamlines)') {
        return {
            modeLabel: 'Chart Mode',
            aggregationLabel: 'Mean Field (HEALPix-style)',
            sliderColorBounds: [-120.0, 120.0],
            patch: {
                ...resetFlipY,
                pipeline: 'cartesian',
                xField: null,
                yField: null,
                colorField: 'radial_velocity',
                colorScale: 'rdbu',
                colorMin: -42.0,
                colorMax: 42.0,
                smoothingKernel: 1,
                invertScale: false,
                lightBackground: true,
                experimental: {
                    aggregation: 'streamlines',
                    metricType: 0,
                    auxFields: ['pmra', 'pmdec', 'parallax'],
                },
            },
        };
    }
    if (lens === 'Hertzsprung-Russell (HR) Diagram') {
        return {
            modeLabel: 'Chart Mode',
            aggregationLabel: 'Additive Points',
            sliderColorBounds: [-1.0, 5.0],
            patch: {
                pipeline: 'cartesian',
                aggregation: 'points',
                xField: 'bp_rp',
                yField: gaiaMagCol,
                flipY: true,
                colorField: 'bp_rp',
                colorScale: 'coolwarm',
                colorMin: -0.5,
                colorMax: 3.5,
                invertScale: false,
                lightBackground: false,
                experimental: {
                    metricType: 0,
                    auxFields: ['', '', ''],
                },
            },
        };
    }

    // 'Custom Chart Mode'
    return {
        modeLabel: 'Chart Mode',
        patch: {
            pipeline: 'cartesian',
            aggregation:
                currentAggregationLabel === 'Mean Field (HEALPix-style)'
                    ? 'mean_field'
                    : 'points',
            experimental: {
                metricType: 0,
                auxFields: ['', '', ''],
            },
        },
    };
}
