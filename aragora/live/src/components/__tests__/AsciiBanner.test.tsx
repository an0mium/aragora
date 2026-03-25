import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AsciiBannerCompact } from '@/components/AsciiBanner';

describe('AsciiBannerCompact', () => {
  it('does not render nested anchors or buttons when wrapped in a link', () => {
    const { container } = render(
      <a href="/">
        <AsciiBannerCompact connected />
      </a>,
    );

    expect(container.querySelectorAll('a')).toHaveLength(1);
    expect(screen.queryByRole('button', { name: /aragora menu/i })).not.toBeInTheDocument();
    expect(screen.getByText('[ARAGORA]')).toBeInTheDocument();
  });

  it('renders an interactive logo only when onLogoClick is provided', async () => {
    const user = userEvent.setup();
    const onLogoClick = jest.fn();

    render(<AsciiBannerCompact onLogoClick={onLogoClick} />);

    await user.click(screen.getByRole('button', { name: /aragora menu/i }));
    expect(onLogoClick).toHaveBeenCalledTimes(1);
  });
});
