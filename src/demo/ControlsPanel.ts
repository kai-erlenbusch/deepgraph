import GUI from 'lil-gui';
import type {
    DatasetMetadataSummary,
    DeepgraphPlot,
    InteractionMode,
    PointHitEvent,
    TelemetrySnapshot,
    ViewOptionsPatch,
} from '../index.ts';
import {
    buildScienceLensPatch,
    GAIA_SCIENCE_LENS_NAMES,
} from './gaiaPreset.ts';

const SPATIAL_X_OPTION = '(Spatial X)';
const SPATIAL_Y_OPTION = '(Spatial Y)';

function ensureLegendElement(): HTMLElement {
    let el = document.getElementById('categorical-legend');
    if (!el) {
        el = document.createElement('div');
        el.id = 'categorical-legend';
        el.style.position = 'absolute';
        el.style.bottom = '20px';
        el.style.right = '20px';
        el.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        el.style.color = 'white';
        el.style.padding = '10px';
        el.style.borderRadius = '5px';
        el.style.fontFamily = 'monospace';
        el.style.fontSize = '12px';
        el.style.display = 'none';
        el.style.pointerEvents = 'none';
        document.body.appendChild(el);
    }
    return el;
}

export function renderCategoricalLegend(
    container: HTMLElement | null,
    fieldName: string,
    dict: string[] | undefined,
    getColorHex: (index: number) => number
): void {
    if (!container) return;
    if (!fieldName || !dict || dict.length === 0) {
        container.style.display = 'none';
        return;
    }
    container.replaceChildren();

    const header = document.createElement('div');
    header.style.fontWeight = 'bold';
    header.style.marginBottom = '5px';
    header.textContent = fieldName;
    container.appendChild(header);

    const maxItems = 30;
    const count = Math.min(dict.length, maxItems);
    for (let i = 0; i < count; i++) {
        const colorHex = getColorHex(i);
        const hexStr = '#' + colorHex.toString(16).padStart(6, '0');

        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.marginBottom = '3px';

        const swatch = document.createElement('span');
        swatch.style.display = 'inline-block';
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.backgroundColor = hexStr;
        swatch.style.marginRight = '8px';
        swatch.style.borderRadius = '2px';

        const label = document.createElement('span');
        label.textContent = String(dict[i]);

        row.appendChild(swatch);
        row.appendChild(label);
        container.appendChild(row);
    }

    if (dict.length > maxItems) {
        const more = document.createElement('div');
        more.style.color = '#aaa';
        more.style.marginTop = '4px';
        more.textContent = `+${dict.length - maxItems} more...`;
        container.appendChild(more);
    }

    container.style.display = 'block';
}

export class TelemetryHud {
    private uiText: HTMLElement;

    constructor(uiText: HTMLElement, copyBtn?: HTMLElement | null) {
        this.uiText = uiText;
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(this.uiText.innerText);
                const origColor = copyBtn.style.color;
                copyBtn.style.color = '#4CAF50';
                setTimeout(() => {
                    copyBtn.style.color = origColor;
                }, 1000);
            });
        }
    }

    public setStatus(message: string, isHtml = false): void {
        if (isHtml) {
            this.uiText.innerHTML = message;
        } else {
            this.uiText.textContent = message;
        }
    }

    public setError(message: string): void {
        this.uiText.replaceChildren();
        const errSpan = document.createElement('span');
        errSpan.style.color = 'red';
        errSpan.style.fontWeight = 'bold';
        errSpan.textContent = `❌ ${message}`;
        this.uiText.appendChild(errSpan);
    }

    public update(snap: TelemetrySnapshot): void {
        const ptsText =
            snap.activePoints < snap.loadedPoints
                ? `${(snap.activePoints / 1_000_000).toFixed(2)}M active / ${(snap.loadedPoints / 1_000_000).toFixed(2)}M loaded`
                : `${(snap.loadedPoints / 1_000_000).toFixed(2)} Million`;

        let sloStatusHtml = '';
        if (snap.sloP95Ms > 0) {
            const color = snap.isSloViolated ? '#ff4444' : '#4CAF50';
            const icon = snap.isSloViolated ? '❌' : '✅';
            sloStatusHtml = `<br/><span style="color:${color}; font-weight: bold;">SLO Status (P95 < ${snap.sloBudgetMs}ms): ${icon} ${snap.sloP95Ms.toFixed(1)}ms</span>`;
        }

        let telZ = document.getElementById('tel-z');
        if (!telZ) {
            this.uiText.innerHTML = `
               Streaming Quadtree (Z=<span id="tel-z">0</span>)
            FPS: <span id="tel-fps">0</span>
            Camera Zoom: <span id="tel-zoom">0</span><br/>
               <span id="tel-pyramid" style="color:#8fd3ff; font-size: 12px;"></span><br/>
               Tiles active: <span id="tel-tiles">0</span> (<span id="tel-vram">0</span> MB tile / <span id="tel-gpu-total">0</span> MB GPU)<br/>
               Points rendered: <span id="tel-pts">0 Million</span><br/>
               Tile load latency: <b>Net</b> <span id="tel-net">0</span>ms | <b>Worker</b> <span id="tel-worker">0</span>ms
               <span id="tel-slo"></span>
               <br/><span style="color:#aaa; font-size: 12px;">Culling: <span id="tel-cull">0</span>ms | GPU Upload: <span id="tel-upload">0</span>ms | Render: <span id="tel-render">0</span>ms</span>
             `;
            telZ = document.getElementById('tel-z');
        }
        if (!telZ) return;

        telZ.textContent = String(snap.currentZ ?? 0);
        const pyrSpan = document.getElementById('tel-pyramid');
        if (pyrSpan && pyrSpan.textContent !== snap.pyramidSummary) {
            pyrSpan.textContent = snap.pyramidSummary;
        }
        document.getElementById('tel-fps')!.textContent = snap.fps.toFixed(0);
        document.getElementById('tel-zoom')!.textContent = snap.cameraZoom.toFixed(2);
        document.getElementById('tel-tiles')!.textContent = String(snap.activeTiles);
        document.getElementById('tel-vram')!.textContent = snap.estimatedVramMB.toFixed(1);
        const totalGpuEl = document.getElementById('tel-gpu-total');
        if (totalGpuEl) {
            totalGpuEl.textContent = (snap.totalGpuMB ?? snap.estimatedVramMB).toFixed(1);
        }
        document.getElementById('tel-pts')!.textContent = ptsText;
        document.getElementById('tel-net')!.textContent = snap.netLatencyMs.toFixed(1);
        document.getElementById('tel-worker')!.textContent = snap.workerLatencyMs.toFixed(1);
        document.getElementById('tel-cull')!.textContent = snap.cullMs.toFixed(1);
        document.getElementById('tel-upload')!.textContent = snap.uploadMs.toFixed(1);
        document.getElementById('tel-render')!.textContent = snap.renderMs.toFixed(1);

        const sloSpan = document.getElementById('tel-slo')!;
        if (sloSpan.innerHTML !== sloStatusHtml) {
            sloSpan.innerHTML = sloStatusHtml;
        }
    }
}

export class ControlsPanel {
    private plot: DeepgraphPlot;
    private gui: GUI | null = null;
    private legendContainer: HTMLElement;
    private trailTimeCtrl: any = null;
    private guiState: any = null;
    private lastLegendKey = '';
    private unsubLoad: (() => void) | null = null;
    private unsubTelemetry: (() => void) | null = null;
    private unsubClick: (() => void) | null = null;
    private unsubHover: (() => void) | null = null;
    private unsubBrush: (() => void) | null = null;
    private inspectEl: HTMLElement | null = null;
    private selectedHit: PointHitEvent | null = null;
    private hoverHit: PointHitEvent | null = null;
    private demoSampledNodes: PointHitEvent[] = [];

    constructor(plot: DeepgraphPlot) {
        this.plot = plot;
        this.legendContainer = ensureLegendElement();
        this.inspectEl = this.ensureInspectElement();

        // Refresh legend whenever `load` fires (including lazy v0.0 tile dictionary arrival)
        this.unsubLoad = this.plot.on('load', () => {
            this.refreshLegend();
            this.selectedHit = null;
            this.hoverHit = null;
            this.demoSampledNodes = [];
            this.renderInspectCard();
        });
        this.unsubClick = this.plot.on('click', (hit) => {
            this.selectedHit = hit;
            this.renderInspectCard();
            this.refreshDemoEgoArcs();
        });
        this.unsubHover = this.plot.on('hover', (hit) => {
            this.hoverHit = hit;
            this.renderInspectCard();
            if (!this.selectedHit) {
                this.refreshDemoEgoArcs();
            }
        });
        this.unsubBrush = this.plot.on('brush', (clause) => {
            if (clause.kind !== 'none' && this.guiState && this.guiState.unselectedOpacity > 0) {
                const cur = this.plot.getFilter();
                if (cur && cur.unselectedOpacity !== this.guiState.unselectedOpacity) {
                    this.plot.setFilter({
                        ...cur,
                        unselectedOpacity: this.guiState.unselectedOpacity,
                    });
                }
            }
        });
        this.unsubTelemetry = this.plot.on('telemetry', (snap) => {
            if (this.guiState?.animateFlow && this.trailTimeCtrl) {
                this.guiState.trailTimeKyr = snap.trailTimeKyr;
                this.trailTimeCtrl.updateDisplay();
            }
            const curField = this.plot.getViewState().colorField;
            const dictLen = curField ? (this.plot.getCategories(curField)?.length ?? 0) : 0;
            const key = `${curField}:${dictLen}`;
            if (key !== this.lastLegendKey) {
                this.refreshLegend();
            }
        });
    }

    public getGui(): GUI | null {
        return this.gui;
    }

    public refreshLegend(): void {
        const vs = this.plot.getViewState();
        const dict = vs.colorField ? this.plot.getCategories(vs.colorField) : undefined;
        this.lastLegendKey = `${vs.colorField}:${dict?.length ?? 0}`;
        renderCategoricalLegend(
            this.legendContainer,
            vs.colorField,
            dict,
            (idx) => this.plot.getCategoryColorHex(idx)
        );
    }

    public rebuild(summary: DatasetMetadataSummary, isGaia: boolean): void {
        if (this.gui) {
            this.gui.destroy();
            this.gui = null;
        }
        this.trailTimeCtrl = null;

        const vs = this.plot.getViewState();
        const schemaFields = summary.schemaFields;
        const colorSizeOptions =
            schemaFields.length > 0 ? schemaFields : [vs.colorField || 'x'];
        const xAxisOptions = [SPATIAL_X_OPTION, ...schemaFields];
        const yAxisOptions = [SPATIAL_Y_OPTION, ...schemaFields];
        const gaiaMagCol = summary.magnitudeField;

        const state = {
            scienceLens: '10. Sky in Colour (Optical Flux)',
            mode: vs.pipeline === 'optical' ? 'Gaia Baseline' : 'Chart Mode',
            detailOffset: vs.detailOffset,
            aggregationMode:
                vs.aggregation === 'points'
                    ? 'Log-Density Points'
                    : 'Mean Field (HEALPix-style)',
            colorField: vs.colorField,
            colorMin: vs.colorMin,
            colorMax: vs.colorMax,
            colorScale: vs.colorScale,
            invertScale: vs.invertScale,
            lightBackground: vs.lightBackground,
            smoothingKernel: vs.smoothingKernel,
            trailStyle: '400 kyr Long Exposure (Poster)',
            trailTimeKyr: vs.experimental.trailTimeKyr,
            minParallax: vs.experimental.minParallax,
            animateFlow: vs.experimental.animateFlow,
            xAxisLabel: vs.xField ?? SPATIAL_X_OPTION,
            yAxisLabel: vs.yField ?? SPATIAL_Y_OPTION,
            sizeField: vs.sizeField,
            sizeMin: vs.sizeMin,
            sizeMax: vs.sizeMax,
            flipY: vs.flipY,
            pointSize: vs.pointSize,
            zoomBalance: vs.zoomBalance,
            autoExposure: vs.exposure === 'auto',
            exposureValue: typeof vs.exposure === 'number' ? vs.exposure : 1.0,
            autoOpacity: vs.pointOpacity === 'auto',
            opacityValue: typeof vs.pointOpacity === 'number' ? vs.pointOpacity : 0.15,
            interactionMode: this.plot.getInteractionMode(),
            unselectedOpacity: this.plot.getFilter()?.unselectedOpacity ?? 0.12,
            demoEgoArcs: false,
            animateArcPulse: false,
            clearSelection: () => {
                this.selectedHit = null;
                this.plot.clearSelection();
                if (!this.guiState?.demoEgoArcs) {
                    this.plot.setEdges([]);
                }
                this.renderInspectCard();
            },
        };
        this.guiState = state;

        let colorMinCtrl: any;
        let colorMaxCtrl: any;
        let sizeMinCtrl: any;
        let sizeMaxCtrl: any;

        const syncSliderRange = (ctrl: any, min: number, max: number) => {
            if (ctrl) ctrl.min(min).max(max);
        };

        const pushStateToPlot = () => {
            const isOpticalBaseline = state.mode === 'Gaia Baseline';
            const curVs = this.plot.getViewState();
            const isSpecialFlow =
                !isOpticalBaseline &&
                isGaia &&
                state.scienceLens !== 'Custom Chart Mode' &&
                (curVs.aggregation === 'trails' || curVs.aggregation === 'streamlines');

            const isCat = Boolean(this.plot.getCategories(state.colorField));
            const baseAgg: 'points' | 'mean_field' =
                !isCat && state.aggregationMode === 'Mean Field (HEALPix-style)'
                    ? 'mean_field'
                    : 'points';

            const patch: ViewOptionsPatch = {
                pipeline: isOpticalBaseline ? 'optical' : 'cartesian',
                detailOffset: state.detailOffset,
                ...(isSpecialFlow ? {} : { aggregation: baseAgg }),
                xField: state.xAxisLabel === SPATIAL_X_OPTION ? null : state.xAxisLabel,
                yField: state.yAxisLabel === SPATIAL_Y_OPTION ? null : state.yAxisLabel,
                colorField: state.colorField,
                colorMin: state.colorMin,
                colorMax: state.colorMax,
                colorScale: state.colorScale,
                invertScale: state.invertScale,
                lightBackground: state.lightBackground,
                smoothingKernel: state.smoothingKernel,
                sizeField: state.sizeField,
                sizeMin: state.sizeMin,
                sizeMax: state.sizeMax,
                flipY: state.flipY,
                pointSize: state.pointSize,
                zoomBalance: state.zoomBalance,
                exposure: state.autoExposure ? 'auto' : state.exposureValue,
                pointOpacity: state.autoOpacity ? 'auto' : state.opacityValue,
                experimental: {
                    metricType:
                        isOpticalBaseline || state.scienceLens === 'Custom Chart Mode' || !isGaia
                            ? 0
                            : curVs.experimental.metricType,
                    auxFields:
                        isOpticalBaseline || state.scienceLens === 'Custom Chart Mode' || !isGaia
                            ? ['', '', '']
                            : curVs.experimental.auxFields,
                    trailMode:
                        state.trailStyle === '1.6 Myr Time Evolution (80 kyr Comets)' ? 1 : 0,
                    trailTimeKyr: state.trailTimeKyr,
                    minParallax: state.minParallax,
                    animateFlow: state.animateFlow,
                },
            };

            this.plot.setViewOptions(patch);
            this.refreshLegend();
        };

        const applyScienceLens = (lens: string) => {
            state.scienceLens = lens;
            const app = buildScienceLensPatch(
                lens,
                gaiaMagCol,
                state.trailStyle,
                state.aggregationMode
            );
            state.mode = app.modeLabel;
            if (app.aggregationLabel) {
                state.aggregationMode =
                    app.aggregationLabel === 'Additive Points'
                        ? 'Log-Density Points'
                        : app.aggregationLabel;
            }
            if (app.patch.flipY !== undefined) state.flipY = app.patch.flipY;
            if (app.patch.xField !== undefined) {
                state.xAxisLabel = app.patch.xField ?? SPATIAL_X_OPTION;
            }
            if (app.patch.yField !== undefined) {
                state.yAxisLabel = app.patch.yField ?? SPATIAL_Y_OPTION;
            }
            if (app.patch.colorField !== undefined) state.colorField = app.patch.colorField;
            if (app.patch.colorScale !== undefined) state.colorScale = app.patch.colorScale;
            if (app.patch.colorMin !== undefined) state.colorMin = app.patch.colorMin;
            if (app.patch.colorMax !== undefined) state.colorMax = app.patch.colorMax;
            if (app.patch.sizeField !== undefined) state.sizeField = app.patch.sizeField;
            if (app.patch.smoothingKernel !== undefined) {
                state.smoothingKernel = app.patch.smoothingKernel;
            }
            if (app.patch.invertScale !== undefined) state.invertScale = app.patch.invertScale;
            if (app.patch.lightBackground !== undefined) {
                state.lightBackground = app.patch.lightBackground;
            }
            if (app.patch.experimental?.trailTimeKyr !== undefined) {
                state.trailTimeKyr = app.patch.experimental.trailTimeKyr;
            }
            if (app.patch.experimental?.minParallax !== undefined) {
                state.minParallax = app.patch.experimental.minParallax;
            }

            if (app.sliderColorBounds) {
                syncSliderRange(
                    colorMinCtrl,
                    app.sliderColorBounds[0],
                    app.sliderColorBounds[1]
                );
                syncSliderRange(
                    colorMaxCtrl,
                    app.sliderColorBounds[0],
                    app.sliderColorBounds[1]
                );
            }

            this.plot.setViewOptions(app.patch);
            if (this.gui) this.gui.controllersRecursive().forEach((c: any) => c.updateDisplay());
            this.refreshLegend();
        };

        const gui = new GUI({ title: 'Visualization Controls' });
        this.gui = gui;

        const availableModes = isGaia ? ['Gaia Baseline', 'Chart Mode'] : ['Chart Mode'];
        const modeFolder = gui.addFolder('Visualization Mode');
        if (isGaia) {
            modeFolder
                .add(state, 'scienceLens', [...GAIA_SCIENCE_LENS_NAMES])
                .name('Gaia Science Lens')
                .onChange(applyScienceLens);
        }
        modeFolder
            .add(state, 'mode', availableModes)
            .name('Mode')
            .onChange((val: string) => {
                if (val === 'Gaia Baseline') {
                    state.scienceLens = '10. Sky in Colour (Optical Flux)';
                } else if (state.scienceLens === '10. Sky in Colour (Optical Flux)') {
                    state.scienceLens = 'Custom Chart Mode';
                }
                if (this.gui) {
                    this.gui.controllersRecursive().forEach((c: any) => c.updateDisplay());
                }
                pushStateToPlot();
            });
        modeFolder
            .add(state, 'detailOffset', 0, 3, 1)
            .name('Z Overfetch')
            .onChange((val: number) => {
                this.plot.setViewOptions({ detailOffset: val });
            });
        modeFolder
            .add({ resetView: () => this.plot.fitView() }, 'resetView')
            .name('Fit View');

        const handleColorFieldChange = (newField: string) => {
            const curVs = this.plot.getViewState();
            if (
                state.scienceLens !== 'Custom Chart Mode' &&
                state.scienceLens !== 'Hertzsprung-Russell (HR) Diagram' &&
                curVs.aggregation !== 'trails'
            ) {
                this.plot.setViewOptions({
                    experimental: { metricType: 0, auxFields: ['', '', ''] },
                });
            }
            const st = summary.stats[newField];
            if (
                st &&
                st.type !== 'categorical' &&
                Number.isFinite(st.min) &&
                Number.isFinite(st.max)
            ) {
                state.colorMin = st.min;
                state.colorMax = st.max;
                syncSliderRange(colorMinCtrl, st.min, st.max);
                syncSliderRange(colorMaxCtrl, st.min, st.max);
                if (this.gui) {
                    this.gui.controllersRecursive().forEach((c: any) => c.updateDisplay());
                }
            }
            pushStateToPlot();
        };

        const handleSizeFieldChange = (newField: string) => {
            const st = summary.stats[newField];
            if (
                st &&
                st.type !== 'categorical' &&
                Number.isFinite(st.min) &&
                Number.isFinite(st.max)
            ) {
                state.sizeMin = st.min;
                state.sizeMax = st.max;
                syncSliderRange(sizeMinCtrl, st.min, st.max);
                syncSliderRange(sizeMaxCtrl, st.min, st.max);
                if (this.gui) {
                    this.gui.controllersRecursive().forEach((c: any) => c.updateDisplay());
                }
            }
            pushStateToPlot();
        };

        const chartFolder = gui.addFolder('Chart & Field Settings');
        chartFolder
            .add(state, 'aggregationMode', ['Mean Field (HEALPix-style)', 'Log-Density Points'])
            .name('Field Blending')
            .onChange(pushStateToPlot);
        chartFolder
            .add(state, 'colorField', colorSizeOptions)
            .name('Color By')
            .onChange(handleColorFieldChange);
        const colorScaleList = isGaia
            ? [
                  'viridis',
                  'plasma',
                  'magma',
                  'inferno',
                  'rdbu',
                  'coolwarm',
                  'jet',
                  'esa_age',
                  'esa_dust',
                  'esa_extinction',
              ]
            : ['viridis', 'plasma', 'magma', 'inferno', 'rdbu', 'coolwarm', 'jet'];
        chartFolder
            .add(state, 'colorScale', colorScaleList)
            .name('Color Scale')
            .onChange(pushStateToPlot);
        chartFolder.add(state, 'invertScale').name('Invert Scale').onChange(pushStateToPlot);
        chartFolder
            .add(state, 'lightBackground')
            .name('Poster Light BG')
            .onChange(pushStateToPlot);
        chartFolder
            .add(state, 'smoothingKernel', 0, 2, 1)
            .name('Pixel Smoothing')
            .onChange(pushStateToPlot);
        chartFolder.add(state, 'flipY').name('Flip Y').onChange(pushStateToPlot);

        let initColorBounds = { min: -1.0, max: isGaia ? 5.0 : 100.0 };
        const initColorStat = summary.stats[state.colorField];
        if (
            initColorStat &&
            initColorStat.type !== 'categorical' &&
            Number.isFinite(initColorStat.min) &&
            Number.isFinite(initColorStat.max)
        ) {
            initColorBounds = { min: initColorStat.min, max: initColorStat.max };
        }
        colorMinCtrl = chartFolder
            .add(state, 'colorMin', initColorBounds.min, initColorBounds.max)
            .name('Color Min')
            .onChange(pushStateToPlot);
        colorMaxCtrl = chartFolder
            .add(state, 'colorMax', initColorBounds.min, initColorBounds.max)
            .name('Color Max')
            .onChange(pushStateToPlot);

        let initSizeBounds = { min: -2.0, max: isGaia ? 20.0 : 100.0 };
        const initSizeStat = summary.stats[state.sizeField];
        if (
            initSizeStat &&
            initSizeStat.type !== 'categorical' &&
            Number.isFinite(initSizeStat.min) &&
            Number.isFinite(initSizeStat.max)
        ) {
            initSizeBounds = { min: initSizeStat.min, max: initSizeStat.max };
        }
        chartFolder
            .add(state, 'sizeField', colorSizeOptions)
            .name('Size By')
            .onChange(handleSizeFieldChange);
        sizeMinCtrl = chartFolder
            .add(state, 'sizeMin', initSizeBounds.min, initSizeBounds.max)
            .name('Size Min')
            .onChange(pushStateToPlot);
        sizeMaxCtrl = chartFolder
            .add(state, 'sizeMax', initSizeBounds.min, initSizeBounds.max)
            .name('Size Max')
            .onChange(pushStateToPlot);

        chartFolder.add(state, 'xAxisLabel', xAxisOptions).name('X Axis').onChange(pushStateToPlot);
        chartFolder.add(state, 'yAxisLabel', yAxisOptions).name('Y Axis').onChange(pushStateToPlot);

        const multiscaleFolder = gui.addFolder('Multiscale & Tone Mapping');
        multiscaleFolder
            .add(state, 'pointSize', 0.25, 3.0, 0.05)
            .name('Point Size')
            .onChange(pushStateToPlot);
        multiscaleFolder
            .add(state, 'zoomBalance', 0.0, 0.6, 0.02)
            .name('Zoom Balance')
            .onChange(pushStateToPlot);
        multiscaleFolder
            .add(state, 'autoExposure')
            .name('Auto Exposure')
            .onChange(pushStateToPlot);
        multiscaleFolder
            .add(state, 'exposureValue', 0.15, 4.0, 0.05)
            .name('Manual Exposure')
            .onChange(() => {
                state.autoExposure = false;
                if (this.gui) {
                    this.gui.controllersRecursive().forEach((c: any) => c.updateDisplay());
                }
                pushStateToPlot();
            });
        multiscaleFolder
            .add(state, 'autoOpacity')
            .name('Auto Opacity')
            .onChange(pushStateToPlot);
        multiscaleFolder
            .add(state, 'opacityValue', 0.01, 1.0, 0.01)
            .name('Manual Opacity')
            .onChange(() => {
                state.autoOpacity = false;
                if (this.gui) {
                    this.gui.controllersRecursive().forEach((c: any) => c.updateDisplay());
                }
                pushStateToPlot();
            });

        const selGraphFolder = gui.addFolder('Selection & Graph');
        selGraphFolder
            .add(state, 'interactionMode', ['pan', 'brush', 'lasso'])
            .name('Pointer Mode')
            .onChange((val: InteractionMode) => {
                this.plot.setInteractionMode(val);
            });
        selGraphFolder
            .add(state, 'unselectedOpacity', 0.0, 0.5, 0.02)
            .name('Unselected Dim')
            .onChange((val: number) => {
                const cur = this.plot.getFilter();
                if (cur) {
                    this.plot.setFilter({ ...cur, unselectedOpacity: val });
                }
            });
        selGraphFolder
            .add(state, 'demoEgoArcs')
            .name('Sample Ego Arcs')
            .onChange(async (enabled: boolean) => {
                if (!enabled) {
                    this.plot.setEdges([]);
                    return;
                }
                await this.ensureDemoSampledNodes();
                this.refreshDemoEgoArcs();
            });
        selGraphFolder
            .add(state, 'animateArcPulse')
            .name('Animate Arc Pulse')
            .onChange((val: boolean) => {
                this.plot.setEdgeStyle({ animatePulse: val });
            });
        selGraphFolder.add(state, 'clearSelection').name('Clear Selection');

        if (isGaia) {
            const motionFolder = gui.addFolder('Astrophysics & Motion');
            motionFolder
                .add(state, 'trailStyle', [
                    '400 kyr Long Exposure (Poster)',
                    '1.6 Myr Time Evolution (80 kyr Comets)',
                ])
                .name('Trail Style')
                .onChange((val: string) => {
                    if (val === '1.6 Myr Time Evolution (80 kyr Comets)') {
                        state.animateFlow = true;
                        if (state.trailTimeKyr === 400) state.trailTimeKyr = 80;
                    } else {
                        state.animateFlow = false;
                        state.trailTimeKyr = 400;
                    }
                    if (this.gui) {
                        this.gui.controllersRecursive().forEach((c: any) => c.updateDisplay());
                    }
                    pushStateToPlot();
                });
            this.trailTimeCtrl = motionFolder
                .add(state, 'trailTimeKyr', 10, 1600, 10)
                .name('Epoch / Horizon (kyr)')
                .onChange(pushStateToPlot);
            motionFolder
                .add(state, 'minParallax', 2.0, 20.0, 0.5)
                .name('Min Parallax (mas)')
                .onChange(pushStateToPlot);
            motionFolder
                .add(state, 'animateFlow')
                .name('Animate (1.6 Myr Loop)')
                .onChange(pushStateToPlot);
        }

        this.refreshLegend();
    }

    public destroy(): void {
        if (this.gui) {
            this.gui.destroy();
            this.gui = null;
        }
        if (this.unsubLoad) this.unsubLoad();
        if (this.unsubTelemetry) this.unsubTelemetry();
        if (this.unsubClick) this.unsubClick();
        if (this.unsubHover) this.unsubHover();
        if (this.unsubBrush) this.unsubBrush();
        if (this.inspectEl) {
            this.inspectEl.remove();
            this.inspectEl = null;
        }
    }

    private async ensureDemoSampledNodes(): Promise<void> {
        if (this.demoSampledNodes.length >= 24) return;
        const canvas = document.querySelector('#app canvas') as HTMLCanvasElement | null;
        if (!canvas) return;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const nodes = new Map<number, PointHitEvent>();
        for (let gx = 0.18; gx <= 0.82; gx += 0.06) {
            for (let gy = 0.14; gy <= 0.86; gy += 0.07) {
                const p = await this.plot.pickAt(w * gx, h * gy);
                if (p && !nodes.has(p.rowId)) {
                    nodes.set(p.rowId, p);
                }
            }
        }
        this.demoSampledNodes = Array.from(nodes.values());
    }

    private refreshDemoEgoArcs(): void {
        if (!this.guiState?.demoEgoArcs) return;
        const hub = this.selectedHit ?? this.hoverHit;
        if (!hub || this.demoSampledNodes.length === 0) {
            this.plot.setEdges([]);
            return;
        }
        const colorCol = this.plot.getViewState().colorField;
        const hubVal = colorCol ? hub.attributes[colorCol] : undefined;
        const matching =
            colorCol && hubVal !== undefined && hubVal !== null
                ? this.demoSampledNodes.filter(
                      (n) => n.rowId !== hub.rowId && n.attributes[colorCol] === hubVal
                  )
                : [];
        const candidates = (
            matching.length >= 3
                ? matching
                : this.demoSampledNodes.filter((n) => n.rowId !== hub.rowId)
        ).slice(0, 32);

        this.plot.setEdges(
            candidates.map((t, idx) => ({
                sourceRowId: hub.rowId,
                targetRowId: t.rowId,
                source: { x: hub.dataX, y: hub.dataY },
                target: { x: t.dataX, y: t.dataY },
                sourceAttributes: hub.attributes,
                targetAttributes: t.attributes,
                weight: 1.0 - (idx / Math.max(1, candidates.length)) * 0.65,
            })),
            {
                width: [0.9, 2.4],
                curvature: 0.18,
                opacity: 0.78,
                animatePulse: Boolean(this.guiState?.animateArcPulse),
            }
        );
    }

    private ensureInspectElement(): HTMLElement {
        let el = document.getElementById('point-inspect-card');
        if (!el) {
            el = document.createElement('div');
            el.id = 'point-inspect-card';
            el.style.position = 'fixed';
            el.style.top = '16px';
            el.style.left = '16px';
            el.style.background = 'rgba(18, 20, 26, 0.90)';
            el.style.backdropFilter = 'blur(8px)';
            el.style.border = '1px solid rgba(255, 209, 56, 0.45)';
            el.style.borderRadius = '8px';
            el.style.padding = '10px 12px';
            el.style.color = '#eee';
            el.style.fontFamily = 'monospace';
            el.style.fontSize = '11px';
            el.style.maxWidth = '320px';
            el.style.zIndex = '1000';
            el.style.display = 'none';
            el.style.pointerEvents = 'none';
            document.body.appendChild(el);
        }
        return el;
    }

    private renderInspectCard(): void {
        if (!this.inspectEl) return;
        const activeHit = this.hoverHit ?? this.selectedHit;
        if (!activeHit) {
            this.inspectEl.style.display = 'none';
            return;
        }
        const isShowingSelected =
            this.selectedHit !== null &&
            (!this.hoverHit || this.hoverHit.rowId === this.selectedHit.rowId);

        this.inspectEl.style.border = isShowingSelected
            ? '1px solid rgba(255, 209, 56, 0.55)'
            : '1px solid rgba(115, 230, 255, 0.55)';
        this.inspectEl.replaceChildren();

        const title = document.createElement('div');
        title.style.fontWeight = 'bold';
        title.style.color = isShowingSelected ? '#ffd138' : '#73e6ff';
        title.style.marginBottom = '4px';
        const idStr =
            activeHit.rowIdBigInt !== undefined
                ? String(activeHit.rowIdBigInt)
                : String(activeHit.rowId);
        title.textContent = isShowingSelected
            ? `Selected row_id: ${idStr}`
            : `Hover row_id: ${idStr}`;
        this.inspectEl.appendChild(title);

        const coords = document.createElement('div');
        coords.style.color = '#8fd3ff';
        coords.style.marginBottom = '4px';
        coords.textContent = `(${Number(activeHit.dataX).toFixed(4)}, ${Number(activeHit.dataY).toFixed(4)})`;
        this.inspectEl.appendChild(coords);

        const entries = Object.entries(activeHit.attributes || {}).slice(0, 8);
        for (const [k, v] of entries) {
            const line = document.createElement('div');
            const valStr =
                typeof v === 'number'
                    ? Number.isInteger(v)
                        ? String(v)
                        : v.toFixed(4)
                    : String(v);
            line.textContent = `${k}: ${valStr}`;
            this.inspectEl.appendChild(line);
        }
        this.inspectEl.style.display = 'block';
    }
}
