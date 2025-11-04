import asyncio
import json
import logging
from typing import Any, List, Literal, Optional

from dotenv import load_dotenv
from jinja2 import Template
from langfuse import observe
from pydantic import BaseModel, Field

from spec_agent.actors import (ActorConfig, Profile, Supervisor, Worker,
                               register_worker)
from spec_agent.main import SpecAgent
from spec_agent.spec import Spec, SubItem, SubTask, SubTaskReview, TaskList

load_dotenv()   

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --- Worker implementation for data visualization ---

@register_worker(
    Profile(actor_name="axis_and_grid", specialty="A worker that improves the axes and grid of a plot function."),
    Profile(actor_name="font_and_text_clarity", specialty="A worker that improves the font and text clarity of a plot function."),
    Profile(actor_name="figure_and_size", specialty="A worker that improves the figure and size of a plot function."),
    Profile(actor_name="color_palette_contrast", specialty="A worker that improves the color palette and contrast of a plot function."),
    Profile(actor_name="lines_markers", specialty="A worker that improves the lines and markers of a plot function."),
    Profile(actor_name="legend_annotations", specialty="A worker that improves the legend and annotations of a plot function."),
    Profile(actor_name="faceting_small_multiples", specialty="A worker that improves the faceting and small multiples usage of a plot function."),
)
class DataVizWorker(Worker):
    config = ActorConfig(
        model="xai/grok-4-fast-reasoning",
    )

    def render_user_prompt(self, subtask: SubTask) -> str:
        user_prompt = Template("""
            Complete the task to the best of your ability without crossing the boundaries of your role, specialty and task.

            Your task is:
            {{ task }}

            Your task context is:
            {{ task_context }}

            """)
        
        return user_prompt.render(task=subtask.task_details, task_context=subtask.task_context)

    @observe(name="spec_agent.worker.perform_work")
    async def perform_work(self, subtask: SubTask, spec: Spec, **_: Any) -> SubTask:
        try:
            context = [
                    {"role": "system", "content": self.render_system_prompt(specialty=self.profile.specialty, answer_format=spec.task_output_format.model_json_schema())},
                    {"role": "user", "content": self.render_user_prompt(subtask=subtask)},
                ] if not subtask.task_context else subtask.task_context + [ {"role": "user", "content": self.render_user_prompt(subtask=subtask)}]
                
            inference_result = await self.llm.acompletion(
                messages=context,
                response_format=spec.task_output_format,
                **self.config.llm_kwargs,
            )

            return SubTask(
                id=subtask.id,
                subitem=subtask.subitem,
                task_details=subtask.task_details,
                task_result=inference_result.content,
                task_cost=inference_result.cost,
                task_context=subtask.task_context + [{"role": "user", "content": inference_result}],
                status="completed",
            )
        except Exception as e:
            raise RuntimeError(f"Error performing work: {str(e)}")

# --- Supervisor that seeds the task and reviews results ---



class DataVizSupervisor(Supervisor):
    config = ActorConfig(
        model="xai/grok-4-fast-reasoning",
    )

    def get_user_prompt_initial_assignment(self, current_plot_function: str) -> str:
        self.user_prompt_initial_assignment = Template("""
                The spec you are working on is:
                {{ spec }}

                The plot function you are working on is:
                {{ current_plot_function }}

                Create a list of tasks to complete the spec.

                """)
        
        return self.user_prompt_initial_assignment.render(spec=str(self.spec), current_plot_function=current_plot_function)

    def get_user_prompt_review(self, subtask: SubTask, current_plot_function: str, is_review:bool, review_result: Optional[Literal["approved", "partially_approved", "rejected"]] = None)-> str:
        # Yes, that's how you use if statements in Jinja2 templates—by using `{% if ... %} ... {% else %} ... {% endif %}` blocks. 
        # However, make sure that when rendering, you actually pass `is_review` and `subtask` as keyword arguments to `.render()`.
        self.user_prompt_review = Template("""
                The spec you are working on is:
                {{ spec }}

                This is the current plot function:
                {{ current_plot_function }}
                
                {% if is_review %}
                The subtask you are reviewing is:
                {{ subtask }}

                Review the subtask and decide whether to approve it, partially approve it or reject it.
                {% else %}

                Last subtask submitted with review result of {{ review_result }}:
                {{ subtask }}

                Propose one (or more) subtask(s) for the same subitem that is being reviewed to improve the subitem acceptance criteria that are not met.
                {% endif %}
                """)
        
        return self.user_prompt_review.render(spec=str(self.spec), subtask=subtask, current_plot_function=current_plot_function, is_review=is_review, review_result=review_result)
    
    @observe(name="spec_agent.first_assignment")
    async def handle_first_assignment(self, *_, **kwargs: Any) -> List[SubTask]:
        try:
            system_prompt = self.render_system_prompt(
                spec_object=self.spec.__class__.model_json_schema(),
                workers=self.get_workers_string(),
                answer_format=self.spec.spec_output_format.model_json_schema()
            )
            user_prompt = self.get_user_prompt_initial_assignment(self.spec.final_result)

            context = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            inference_result = await self.llm.acompletion(
                messages=context,
                response_format=TaskList,
                **self.config.llm_kwargs,
            )
            
            # Track supervisor cost
            self.spec_cost += inference_result.cost

            initial_tasks = []
            for task in inference_result.content.tasks:
                initial_tasks.append(SubTask(
                    subitem=self.spec.all_subitems[task.assigned_subitem_type],
                    task_details=task,
                    status="pending",
                    task_error=None,
                    task_result=None,
                    task_context=[],
                ))
            return initial_tasks
        
        except Exception:
            raise RuntimeError("Error generating initial tasks")

    @observe(name="spec_agent.review")
    async def review(self, subtask: SubTask, *_, **kwargs: Any) -> List[SubTask]:
        try:
            # review process
            system_prompt = self.render_system_prompt(spec_object=self.spec.__class__.model_json_schema(), workers=self.get_workers_string())
            
            review_result = await self.llm.acompletion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.get_user_prompt_review(subtask=subtask, current_plot_function=self.spec.final_result, is_review=True, review_result=None)},
                ],
                response_format=SubTaskReview,
                **self.config.llm_kwargs,
            )
            
            # Track supervisor cost
            self.spec_cost += review_result.cost
            
            if not review_result.content:
                raise ValueError("Review result is required")

            # Merge phase
            if review_result.content.review_result in ["approved", "partially_approved"]:
                merge_system_prompt = """
                You will be given a initial plotion function and a new code diff that improves the plot function.

                Make sure you analyze the diff and merge it into the initial plot function in the most appropriate way.
                
                Special case:
                There might be differences between the diff base function and the initial plot function, as potential changes might have been made after the diff was genreted. 
                In those cases try to extract the intention of the diff and merge it into the initial plot function in way that addresses the intended changes.
                Even though the root function of the diff might be out of sync, these diffs are supposed to be applied to very specific parts of the initial plot function and address specific functions of it. Make sure you capture that.

                Make sure you return a valid plot function that takes a pandas dataframe and returns a svg string. Pay extreme attetion to identation so I can copy and paste it into the code editor.
                """
                merge_prompt = Template("""
                The initial plot function is:
                {{ initial_plot_function }}

                The diff is:
                {{ diff }}
                """)

                inference_result = await self.llm.acompletion(
                    messages=[
                        {"role": "system", "content": merge_system_prompt},
                        {"role": "user", "content": merge_prompt.render(initial_plot_function=self.spec.final_result, diff=subtask.task_result)},
                    ],
                    response_format=self.spec.spec_output_format,
                    **self.config.llm_kwargs,
                )
                
                # Track supervisor cost
                self.spec_cost += inference_result.cost
                
                self.spec = self.spec.merge_review(subtask, final_result=inference_result.content)


            # New task assignment phase
            
            if review_result.content.review_result == "approved":
                return []
            else:
                system_prompt = self.render_system_prompt(spec_object=self.spec.__class__.model_json_schema(), workers=self.get_workers_string())
                user_prompt = self.get_user_prompt_review(current_plot_function=self.spec.final_result, subtask=subtask, is_review=True, review_result=review_result.content.review_result)
                context = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                inference_result = await self.llm.acompletion(
                    messages=context,
                    response_format=TaskList,
                    **self.config.llm_kwargs,
                )
                
                # Track supervisor cost
                self.spec_cost += inference_result.cost

                new_tasks = []
                for task in inference_result.content.tasks:
                    new_tasks.append(SubTask(
                        subitem=self.spec.all_subitems[task.assigned_subitem_type],
                        task_details=task,
                        status="pending",
                        task_error=None,
                        task_result=None,
                        task_context=context + [{"role": "assistant", "content": f"A new task to complete the spec has been proposed: {str(task)}"}],
                    ))
                return new_tasks

            
        except Exception as e:
            raise RuntimeError(f"Error reviewing subtask: {str(e)}")
        

# --- Spec definition ---

class SpecOutputFormat(BaseModel):
    plot_function: str = Field(description="The plot function that takes a pandas dataframe and returns a svg string.")

class TaskOutputFormat(BaseModel):
    plot_function_diff: str = Field(description="The diff of the plot function. This is a string that describes the diff of the plot function.")

VizSpec = Spec(
    title="Improve the visualization",
    description="Given a current plot function, improve the plot so it mets a specific set of criteria in terms of styling and layout",
    acceptance_criteria="A plot function that takes a pandas dataframe and returns a svg string. Take the set of characteristics: axes and grid, font and text clarity, figure and size, color palette and contrast, lines and markers, legend and annotations, and faceting and small multiples and improve the plot so it meets the criteria.",
    subitems_type={"axis_and_grid", "font_and_text_clarity", "figure_and_size", "color_palette_contrast", "lines_markers", "legend_annotations", "faceting_small_multiples"},
    all_subitems={
        "axis_and_grid": SubItem(
            type="axis_and_grid",
            description="Improve the axes and grid of the plot",
            acceptance_criteria="Creates clean, unobtrusive axis and grid styling. MUST include: (1) Spine management: removes top and right spines (ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)) OR uses a style that does this automatically; (2) Remaining spines styled subtly: uses gray color (#666666 or similar) instead of black, with moderate linewidth (0.8-1.2pt); (3) Grid implementation: if grids are used, they are light gray (#CCCCCC or lighter), thin (linewidth ≤1.0), and low opacity (alpha ≤0.5); (4) Background choice: white for print contexts, or consistent neutral color that doesn't compete with data; (5) Tick styling: outward-facing ticks preferred, reasonable size (not too long/short), adequate spacing from labels; (6) Labeled axsis with the quantity and its unit in parethesis for cleaner look and easy of use.",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
        "font_and_text_clarity": SubItem(
            type="font_and_text_clarity",
            description="Improve the font and text clarity of the plot",
            acceptance_criteria="Implements consistent, readable typography throughout. MUST include: (1) Single font family used across all text elements: serif (Times New Roman, PT Serif) for academic contexts OR sans-serif (Arial, Lato) for presentations; (2) Appropriate font size hierarchy: title 12-16pt, axis labels 10-12pt, tick labels 8-10pt, legend 8-10pt, set via rcParams or individual element configuration; (3) Sufficient size for readability: text remains legible when figure is viewed at intended final size; (4) Mathematical text consistency: uses same font family for math expressions OR ensures math font (Computer Modern, STIX) harmonizes with main text; (5) No font mixing: avoids combining serif and sans-serif within same plot; (6) Proper text positioning: adequate spacing between text elements, no overlapping labels. All text should appear intentionally formatted rather than using matplotlib defaults.",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
        "figure_and_size": SubItem(
            type="figure_and_size",
            description="Improve the figure and size of the plot",
            acceptance_criteria="Creates appropriately sized and well-organized figure layout. MUST include: (1) Intentional figure dimensions: uses plt.figure(figsize=(width, height)) with reasonable proportions for the intended use (e.g., 6-8 inches wide for presentations, 3-4 inches for single-column papers); (2) Proper aspect ratio: chooses aspect ratio appropriate for data type (wider for time series, square for correlations, equal aspect for geographic data); (3) Layout management: uses plt.tight_layout() or plt.subplots_adjust() to prevent text cutoffs and ensure adequate spacing; (4) Margin control: provides sufficient space around plot elements so nothing appears cramped or cut off; (5) Subplot coordination: if multiple panels present, they are aligned and consistently sized with appropriate spacing; (6) Text scaling: all text elements remain readable at the chosen figure size; (7) No wasted space: figure dimensions are efficient without excessive empty areas. The layout should appear planned and professional rather than using matplotlib defaults.",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
        "color_palette_contrast": SubItem(
            type="color_palette_contrast",
            description="Improve the color palette and contrast of the plot",
            acceptance_criteria="Uses a deliberate, professional color system. MUST include: (1) For categorical data: uses matplotlib's 'Set2', 'Dark2', 'Pastel1' colormaps OR the specific muted palette ['#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#CC6677', '#AA4499', '#DDDDDD'] OR avoids tab10 in favor of better alternatives; (2) For continuous data: uses perceptually uniform colormaps ('viridis', 'plasma', 'inferno', 'cividis', 'magma') instead of 'jet' or 'rainbow'; (3) Adequate contrast: dark colors on light backgrounds OR light colors on dark backgrounds, no low-contrast combinations; (4) Consistent application: same color represents same data category throughout the plot; (5) Limited palette: uses ≤6 distinct colors for categorical data to avoid confusion; (6) Color-blind consideration: avoids problematic red-green combinations or provides alternative visual encodings. Colors should appear intentionally chosen rather than using defaults.",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
        "lines_markers": SubItem(
            type="lines_markers",
            description="Improve the lines and markers of the plot",
            acceptance_criteria="Renders data elements with clear, professional styling. MUST include: (1) Line weights: uses linewidth=1.2-2.0 (thicker than matplotlib default of ~1.0) for primary data lines; (2) Line differentiation: when multiple lines present, uses different styles (solid '-', dashed '--', dotted ':', dashdot '-.') not just colors; (3) Marker specifications: uses distinct marker shapes ('o', 's', '^', 'D') with reasonable size (markersize=5-8), large enough to see but not overwhelming; (4) Marker visibility: filled markers have subtle edge colors (markeredgecolor) or sufficient contrast with background; (5) Bar/area styling: bars have light edge colors or outlines to define boundaries clearly; (6) Overplotting management: uses alpha transparency (0.6-0.8) when points overlap, or reduces marker size in dense plots; (7) Smooth rendering: lines appear anti-aliased and smooth, not pixelated. Data elements should be easily distinguishable and readable at the target output size.",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
        "legend_annotations": SubItem(
            type="legend_annotations",
            description="Improve the legend and annotations of the plot",
            acceptance_criteria="Implements clear, well-positioned legend and annotation system. MUST include: (1) Legend placement: positioned to avoid data overlap, typically outside plot area or in empty corner using loc parameter ('upper right', 'lower left', etc.); (2) Legend styling: font size matches or is slightly smaller than axis labels (fontsize=8-11), with readable but not oversized symbols; (3) Legend content: concise, descriptive labels that clearly identify each data series (without creating too many categories - if that is need should break into subplots); (4) Legend frame: if present, uses subtle styling (framealpha=0.8-1.0, light edge color) that doesn't dominate; (5) Annotation usage: when present, annotations are purposeful and limited (≤3 per plot), with consistent styling and readable text; (6) Annotation positioning: placed to avoid data overlap, with clear connection to referenced data points.",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
        "faceting_small_multiples": SubItem(
            type="faceting_small_multiples",
            description="Improve the faceting and small multiples usage of the plot",
            acceptance_criteria="Makes appropriate decision to use small multiples when beneficial. MUST include: (1) Justified faceting: uses subplots when there are >3-4 data series, different units/scales, or when patterns are clearer when separated; (2) Consistent formatting: all subplots use same styling, color schemes, and axis formatting; (3) Proper spacing: uses hspace/wspace parameters or plt.tight_layout() to ensure adequate separation between panels; (4) Clear panel identification: each subplot has clear titles or labels to identify what data subset it represents. The faceted design should make comparisons easier, not more confusing.",
            task_output_format=json.dumps(TaskOutputFormat.model_json_schema()),
        ),
    },
    spec_output_format=SpecOutputFormat,
    task_output_format=TaskOutputFormat,
)
# --- task assets --- 

data = [
    {
        "Accurate": "82.0",
        "Whole sample": "Method Alpha",
        "Overestimation": "5.0",
        "Underestimation": "13.0",
    },
    {
        "Accurate": "74.0",
        "Whole sample": "Method Beta",
        "Overestimation": "0.0",
        "Underestimation": "26.0",
    },
    {
        "Accurate": "71.0",
        "Whole sample": "Method Gamma",
        "Overestimation": "11.0",
        "Underestimation": "18.0",
    },
    {
        "Accurate": "69.0",
        "Whole sample": "Method Delta",
        "Overestimation": "14.0",
        "Underestimation": "17.0",
    },
    {
        "Accurate": "43.0",
        "Whole sample": "Method Epsilon",
        "Overestimation": "0.0",
        "Underestimation": "57.0",
    },
]


CURRENT_PLOT_FUNCTION = """
def create_plot(df):`
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import io

    # Set style and rcParams
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # Data preparation
    # Assume 'Whole sample' is the method names, others are percentages as strings
    df = df.copy()
    method_col = "Whole sample"
    methods = df[method_col].tolist()
    cat_cols = ["Accurate", "Overestimation", "Underestimation"]

    # Convert to float
    for col in cat_cols:
        df[col] = df[col].astype(float)

    # Calculate totals
    totals = df[cat_cols].sum(axis=1)
    zero_mask = totals == 0

    # Normalize to 100%
    normalized_df = df[cat_cols].div(totals, axis=0) * 100
    normalized_df = normalized_df.fillna(0)

    # Original for tooltips and bias
    original_percentages = df[cat_cols]

    # Sort by Accurate descending
    sort_idx = df["Accurate"].argsort()[::-1]
    df_sorted = df.iloc[sort_idx].reset_index(drop=True)
    normalized_df_sorted = normalized_df.iloc[sort_idx].reset_index(drop=True)
    methods_sorted = df_sorted[method_col].tolist()
    original_sorted = original_percentages.iloc[sort_idx].reset_index(drop=True)
    zero_totals_sorted = zero_mask.iloc[sort_idx].reset_index(drop=True)

    # Compute net bias from original
    net_bias = df_sorted["Overestimation"] - df_sorted["Underestimation"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors and hatches
    colors = {
        "Accurate": "#0072B2",
        "Overestimation": "#D55E00",
        "Underestimation": "#009E73",
    }
    hatches = {"Underestimation": "/", "Accurate": "", "Overestimation": "\\\\"}
    order = [
        "Underestimation",
        "Accurate",
        "Overestimation",
    ]  # Stacking order bottom to top

    y_pos = np.arange(len(methods_sorted))

    # Function to get text color
    def get_text_color(hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        avg = (r + g + b) / 3
        return "white" if avg < 128 else "black"

    for i, method in enumerate(methods_sorted):
        if zero_totals_sorted.iloc[i]:
            # Handle zero total
            ax.barh(i, 100, height=0.8, color="lightgray", alpha=0.5, hatch="//")
            ax.text(
                50,
                i,
                "Data missing",
                ha="center",
                va="center",
                color="red",
                fontsize=9,
                fontweight="bold",
            )
            # Still add net bias if computable, but since zero, maybe 0
            row_net = 0
            bias_color = "black"
            ax.text(
                102,
                i,
                f"{row_net:+.1f}",
                ha="left",
                va="center",
                color=bias_color,
                fontsize=10,
                fontweight="bold",
            )
            continue

        row_normalized = normalized_df_sorted.iloc[i]
        original_row = original_sorted.iloc[i]
        row_net = net_bias.iloc[i]

        left = 0
        for cat in order:
            width = row_normalized[cat]
            if width > 0:
                # Bar segment
                rect = ax.barh(
                    i,
                    width,
                    left=left,
                    height=0.8,
                    color=colors[cat],
                    hatch=hatches[cat],
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=1.0,
                )
                left += width

                # Annotation
                pct = row_normalized[cat]
                if pct > 6:
                    # Inside
                    text_color = get_text_color(colors[cat])
                    ax.text(
                        left - width / 2,
                        i,
                        f"{pct:.1f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                        fontsize=9,
                    )
                else:
                    # Outside with leader
                    ax.annotate(
                        f"{pct:.1f}%",
                        xy=(left, i),
                        xytext=(left + 2, i),
                        arrowprops=dict(arrowstyle="->", lw=0.5, color="gray"),
                        ha="left",
                        va="center",
                        fontsize=9,
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=0.5
                        ),
                    )

        # Net bias annotation
        bias_color = "red" if row_net > 0 else "green" if row_net < 0 else "black"
        ax.text(
            102,
            i,
            f"{row_net:+.1f}",
            ha="left",
            va="center",
            color=bias_color,
            fontsize=10,
            fontweight="bold",
        )

    # Axis setup
    ax.set_xlim(0, 105)  # Extra space for bias
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Percentage Composition")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods_sorted)
    ax.set_ylabel("Methods")

    # Spines and grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5, color="#CCCCCC")

    # Legend
    legend_elements = [
        Patch(facecolor=colors[cat], hatch=hatches[cat], label=cat) for cat in order
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))

    # Footnote
    fig.text(
        0.5,
        0.02,
        "Each bar shows the composition for that method (row) — categories are not aggregated across methods.",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    # Export to SVG
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=100)
    buf.seek(0)
    svg_string = buf.read().decode("utf-8")
    plt.close(fig)
    return svg_string
"""

@observe(name="spec_agent")
async def main() -> None:
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    agent = SpecAgent(supervisor=DataVizSupervisor())
    completed_spec = await agent.complete_spec(spec=VizSpec, spec_output_format=SpecOutputFormat, task_output_format=TaskOutputFormat, goal_output=SpecOutputFormat(plot_function=CURRENT_PLOT_FUNCTION), data_frame=data, max_rounds=25)

    from rich import print as rprint
    print(" ===== FINAL FUNCTION ===== \n")
    rprint(completed_spec.spec.final_result['plot_function'])

    print(" ===== COST BREAKDOWN ===== \n")
    rprint(f"Total Cost: ${completed_spec.total_cost:.4f}")
    rprint(f"Supervisor Cost: ${completed_spec.supervisor_cost:.4f}")
    rprint(f"Worker Cost: ${completed_spec.worker_cost:.4f}")

    print("\n ===== FINAL SPEC ===== \n")
    rprint(completed_spec.spec.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())


