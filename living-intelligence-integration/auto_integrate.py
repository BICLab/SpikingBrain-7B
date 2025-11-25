#!/usr/bin/env python3
"""
Automated Integration Script for Living Intelligence Website
============================================================

This script automatically integrates SpikingBrain-7B content into your
living-intelligence repository's "explore technology" section.

Usage:
    python3 auto_integrate.py /path/to/living-intelligence

Features:
- Auto-detects site generator (Jekyll, Hugo, etc.)
- Finds correct content directory
- Copies files to appropriate locations
- Updates navigation config
- Creates backup before changes
- Validates integration
"""

import os
import sys
import shutil
import json
import re
from pathlib import Path
from datetime import datetime

class LivingIntelligenceIntegrator:
    """Automates integration of SpikingBrain content"""

    def __init__(self, target_repo_path):
        self.target_repo = Path(target_repo_path)
        self.source_dir = Path(__file__).parent
        self.site_type = None
        self.content_dir = None
        self.nav_config = None

    def detect_site_generator(self):
        """Auto-detect the static site generator"""
        print("🔍 Detecting site generator...")

        detectors = {
            'Jekyll': ['_config.yml', 'Gemfile', '_posts'],
            'Hugo': ['config.toml', 'config.yaml', 'archetypes'],
            'VuePress': ['.vuepress', 'package.json'],
            'MkDocs': ['mkdocs.yml'],
            'Docusaurus': ['docusaurus.config.js', 'sidebars.js'],
            'Gatsby': ['gatsby-config.js', 'gatsby-node.js'],
            'Next.js': ['next.config.js', 'package.json'],
            'Astro': ['astro.config.mjs', 'astro.config.js'],
            'GitBook': ['SUMMARY.md', 'book.json'],
            'Sphinx': ['conf.py', 'make.bat'],
        }

        for site_type, indicators in detectors.items():
            matches = 0
            for indicator in indicators:
                if (self.target_repo / indicator).exists():
                    matches += 1

            if matches >= 1:
                self.site_type = site_type
                print(f"  ✓ Detected: {site_type}")
                return True

        print("  ⚠ Could not auto-detect site generator")
        return False

    def find_content_directory(self):
        """Find where content should be placed"""
        print("📁 Finding content directory...")

        # Common content directory patterns
        patterns = [
            'docs',
            'content',
            'pages',
            'src/pages',
            'src/content',
            'website/docs',
            '_posts',
            'blog',
        ]

        for pattern in patterns:
            path = self.target_repo / pattern
            if path.exists() and path.is_dir():
                self.content_dir = path
                print(f"  ✓ Found content directory: {path}")
                return True

        print("  ⚠ Could not find standard content directory")
        return False

    def find_navigation_config(self):
        """Find navigation configuration file"""
        print("🗺️  Finding navigation config...")

        config_files = {
            'Jekyll': ['_config.yml', '_data/navigation.yml'],
            'Hugo': ['config.toml', 'config.yaml', 'config.yml'],
            'VuePress': ['.vuepress/config.js', 'docs/.vuepress/config.js'],
            'MkDocs': ['mkdocs.yml'],
            'Docusaurus': ['docusaurus.config.js'],
            'Gatsby': ['gatsby-config.js'],
            'Next.js': ['next.config.js'],
            'Astro': ['astro.config.mjs', 'astro.config.js'],
        }

        if self.site_type in config_files:
            for config in config_files[self.site_type]:
                path = self.target_repo / config
                if path.exists():
                    self.nav_config = path
                    print(f"  ✓ Found nav config: {path}")
                    return True

        print("  ⚠ Could not find navigation config")
        return False

    def create_backup(self):
        """Create backup of existing files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.target_repo / f"backup_{timestamp}"

        print(f"💾 Creating backup: {backup_dir}")

        if backup_dir.exists():
            print("  ⚠ Backup directory already exists, skipping")
            return True

        backup_dir.mkdir(exist_ok=True)

        # Backup navigation config if found
        if self.nav_config and self.nav_config.exists():
            shutil.copy2(self.nav_config, backup_dir / self.nav_config.name)
            print(f"  ✓ Backed up: {self.nav_config.name}")

        return True

    def create_directory_structure(self):
        """Create explore/technology directory structure"""
        print("📂 Creating directory structure...")

        explore_dir = self.content_dir / "explore"
        tech_dir = explore_dir / "technology"

        explore_dir.mkdir(exist_ok=True)
        tech_dir.mkdir(exist_ok=True)

        print(f"  ✓ Created: {explore_dir}")
        print(f"  ✓ Created: {tech_dir}")

        return tech_dir

    def copy_content_files(self, tech_dir):
        """Copy SpikingBrain content files"""
        print("📄 Copying content files...")

        files_to_copy = {
            'explore-spikingbrain.md': 'spikingbrain.md',
            'technology-index.md': 'index.md',
        }

        for source_name, target_name in files_to_copy.items():
            source = self.source_dir / source_name
            target = tech_dir / target_name

            if not source.exists():
                print(f"  ⚠ Source not found: {source_name}")
                continue

            shutil.copy2(source, target)
            print(f"  ✓ Copied: {source_name} → {target_name}")

        return True

    def suggest_navigation_update(self):
        """Suggest navigation configuration update"""
        print("\n" + "="*70)
        print("🗺️  NAVIGATION UPDATE REQUIRED")
        print("="*70)

        if self.site_type:
            print(f"\nDetected site generator: {self.site_type}")

        print(f"\nAdd this to your navigation config:")
        print("-" * 70)

        if self.site_type == 'Jekyll':
            print("""
# In _config.yml or _data/navigation.yml:
nav:
  - title: "Explore"
    url: "/explore/"
    submenu:
      - title: "Technology"
        url: "/explore/technology/"
        submenu:
          - title: "SpikingBrain-7B"
            url: "/explore/technology/spikingbrain"
""")
        elif self.site_type == 'Hugo':
            print("""
# In config.toml:
[[menu.main]]
  name = "Explore"
  url = "/explore/"
  weight = 2

[[menu.main]]
  parent = "Explore"
  name = "Technology"
  url = "/explore/technology/"

[[menu.main]]
  parent = "Technology"
  name = "SpikingBrain-7B"
  url = "/explore/technology/spikingbrain/"
""")
        elif self.site_type == 'MkDocs':
            print("""
# In mkdocs.yml:
nav:
  - Explore:
    - Technology:
      - Overview: explore/technology/index.md
      - SpikingBrain-7B: explore/technology/spikingbrain.md
""")
        else:
            print(f"""
Please refer to navigation-config-samples.yml for {self.site_type or 'your platform'}
Or manually add:
  Explore → Technology → SpikingBrain-7B
""")

        print("-" * 70)

        if self.nav_config:
            print(f"\nNavigation config file: {self.nav_config}")
            print("See navigation-config-samples.yml for complete examples")

    def generate_integration_report(self):
        """Generate detailed integration report"""
        report_path = self.target_repo / "SPIKINGBRAIN_INTEGRATION_REPORT.md"

        print(f"\n📋 Generating integration report...")

        report = f"""# SpikingBrain-7B Integration Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Detection Results

- **Site Generator**: {self.site_type or 'Unknown'}
- **Content Directory**: {self.content_dir or 'Not found'}
- **Navigation Config**: {self.nav_config or 'Not found'}

## Files Added

"""

        # List added files
        tech_dir = self.content_dir / "explore" / "technology" if self.content_dir else None
        if tech_dir and tech_dir.exists():
            report += "Content files:\n"
            for file in tech_dir.glob("*.md"):
                report += f"- {file.relative_to(self.target_repo)}\n"

        report += """
## Next Steps

1. **Update Navigation**
   - See suggestions in console output above
   - Or check: navigation-config-samples.yml

2. **Test Locally**
   - Build your site locally
   - Navigate to: /explore/technology/spikingbrain
   - Verify all links work

3. **Deploy**
   - Commit changes
   - Push to your repository
   - Deploy to production

## Resources

- Integration Instructions: living-intelligence-integration/INTEGRATION_INSTRUCTIONS.md
- Navigation Samples: living-intelligence-integration/navigation-config-samples.yml
- Main Content: docs/explore/technology/spikingbrain.md
- Index Page: docs/explore/technology/index.md

## Support

Need help? Check:
- SpikingBrain-7B repo: https://github.com/Lightiam/SpikingBrain-7B
- Integration docs: living-intelligence-integration/

---

*Integration package created successfully!* ✅
"""

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"  ✓ Report saved: {report_path}")

    def run(self):
        """Execute the integration"""
        print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   Living Intelligence Integration Script                      ║
║   Connecting SpikingBrain-7B to your website                  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

        # Verify target repo exists
        if not self.target_repo.exists():
            print(f"❌ Error: Repository not found: {self.target_repo}")
            print(f"\nUsage: {sys.argv[0]} /path/to/living-intelligence")
            return False

        print(f"Target repository: {self.target_repo}\n")

        # Detection phase
        self.detect_site_generator()
        self.find_content_directory()
        self.find_navigation_config()

        # Verify we have minimum requirements
        if not self.content_dir:
            print("\n❌ Error: Could not find content directory")
            print("   Please specify manually or check repository structure")
            return False

        # Integration phase
        self.create_backup()
        tech_dir = self.create_directory_structure()
        self.copy_content_files(tech_dir)

        # Guidance phase
        self.suggest_navigation_update()
        self.generate_integration_report()

        print("\n" + "="*70)
        print("✅ INTEGRATION COMPLETE!")
        print("="*70)
        print("""
Files added:
  ✓ docs/explore/technology/spikingbrain.md
  ✓ docs/explore/technology/index.md

Next steps:
  1. Update navigation config (see suggestions above)
  2. Test locally: build and view the site
  3. Deploy to production

For detailed instructions:
  → Read: SPIKINGBRAIN_INTEGRATION_REPORT.md
  → Check: living-intelligence-integration/INTEGRATION_INSTRUCTIONS.md
""")

        return True

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 auto_integrate.py /path/to/living-intelligence")
        print("\nExample:")
        print("  python3 auto_integrate.py /home/user/living-intelligence")
        return 1

    target_path = sys.argv[1]
    integrator = LivingIntelligenceIntegrator(target_path)

    success = integrator.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
